"""Offline cost and correctness regressions using the real shared client boundary."""
import json
from unittest.mock import Mock

import pytest

from llm import watsonx_client as client
from llm.json_call import json_call
from llm.usage import track_usage
from reasoning.models import AskRequest, Decomposition
from reasoning import pipeline

DONE = {'sufficient': True, 'reason': 'Covered', 'query': None, 'missing_fact': None}
MISSING = {'sufficient': False, 'reason': 'Missing trigger', 'query': 'retrieval trigger', 'missing_fact': 'Retrieval trigger'}
ANSWER = {'claims': [{'claim_id': 'a', 'text': 'The method retrieves two passages.', 'citation_ids': ['c1']}],
          'abstained': False, 'limitations': []}
AUDIT = {'verdicts': [{'claim_id': 'a', 'status': 'supported', 'explanation': 'The passage states two passages.', 'evidence_ids': ['c1']}]}


class Retriever:
    def __init__(self, evidence):
        self.evidence = evidence
        self.version = 'v1'
        self.queries = []

    def search(self, query, top_k):
        self.queries.append(query)
        return [self.evidence]

    def fingerprint(self):
        return self.version


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    pipeline._cache.clear()
    monkeypatch.setattr(client, '_quota_exhausted', False)
    monkeypatch.setenv('LLM_USAGE_LOG', str(tmp_path / 'usage.jsonl'))
    monkeypatch.setenv('WATSONX_MAX_NEW_TOKENS', '3000')


@pytest.fixture
def provider(monkeypatch):
    model = Mock()
    def install(values):
        def responses():
            for value in values:
                yield {'choices': [{'message': {'content': value if isinstance(value, str) else json.dumps(value)}}],
                       'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}
        model.chat.side_effect = responses()
        monkeypatch.setattr(client, '_model', lambda: model)
        return model
    return install


def test_default_two_calls_and_exact_usage(provider, evidence):
    model = provider([{'answer': ANSWER, 'decision': DONE}, {'critique': AUDIT, 'final': ANSWER}])
    result = pipeline.run_pipeline(AskRequest(question='How does it work?'), Retriever(evidence))
    assert result.mode == 'efficient' and result.final.claims
    assert result.usage['model_calls'] == model.chat.call_count == 2
    assert result.usage['prompt_tokens'] == 200 and result.usage['completion_tokens'] == 40
    assert all(c.kwargs['params']['max_tokens'] == 1800 for c in model.chat.call_args_list)
    assert len(result.critique_log) == 1 and not result.revised
    assert any('separate second audit' in w for w in result.warnings)


def test_cache_reuses_only_same_question_collection_corpus_and_settings(provider, evidence, monkeypatch):
    values = [{'answer': ANSWER, 'decision': DONE}, {'critique': AUDIT, 'final': ANSWER}] * 6
    model = provider(values)
    retriever = Retriever(evidence)
    request = AskRequest(question='How does it work?')
    original = pipeline.run_pipeline(request, retriever)
    cached = pipeline.run_pipeline(request, retriever)
    assert cached.cache_hit and cached.usage['model_calls'] == 0 and cached.usage['prompt_tokens'] == 0
    cached.final.claims.clear()
    assert pipeline.run_pipeline(request, retriever).final.claims  # no mutation leaks
    assert original.usage['model_calls'] == 2
    retriever.version = 'v2'
    assert not pipeline.run_pipeline(request, retriever).cache_hit
    monkeypatch.setenv('WATSONX_MODEL_ID', 'another-model')
    assert not pipeline.run_pipeline(request, retriever).cache_hit
    assert not pipeline.run_pipeline(request.model_copy(update={'collection_id': 'a' * 32}), retriever).cache_hit
    assert not pipeline.run_pipeline(request.model_copy(update={'question': 'What is the method?'}), retriever).cache_hit
    assert not pipeline.run_pipeline(request.model_copy(update={'use_cache': False}), retriever).cache_hit
    assert model.chat.call_count == 12


def test_one_followup_keeps_context_and_uses_three_calls(provider, evidence):
    class Followup(Retriever):
        def search(self, query, top_k):
            self.queries.append(query)
            return [self.evidence if len(self.queries) == 1 else self.evidence.model_copy(update={'chunk_id': 'c2'})]
    model = provider([{'answer': ANSWER, 'decision': MISSING}, {'answer': ANSWER, 'decision': DONE},
                      {'critique': AUDIT, 'final': ANSWER}])
    retriever = Followup(evidence)
    result = pipeline.run_pipeline(AskRequest(question='How does it work?', max_hops=3), retriever)
    assert model.chat.call_count == 3 and len(result.hop_log) == 2
    assert retriever.queries == ['How does it work?', 'retrieval trigger']
    assert {e.chunk_id for e in result.evidence} == {'c1', 'c2'}


def test_no_new_evidence_avoids_another_draft_call(provider, evidence):
    model = provider([{'answer': ANSWER, 'decision': MISSING}, {'critique': AUDIT, 'final': ANSWER}])
    result = pipeline.run_pipeline(AskRequest(question='How does it work?'), Retriever(evidence))
    assert model.chat.call_count == 2
    assert result.hop_log[-1]['stop_reason'] == 'no_new_evidence'
    assert any('Retrieval trigger' in w for w in result.warnings)


def test_review_cannot_keep_unchanged_flagged_claim(provider, evidence):
    bad_audit = {'verdicts': [{**AUDIT['verdicts'][0], 'status': 'unsupported'}]}
    provider([{'answer': ANSWER, 'decision': DONE}, {'critique': bad_audit, 'final': ANSWER}])
    result = pipeline.run_pipeline(AskRequest(question='How does it work?'), Retriever(evidence))
    assert result.final.abstained and result.revised


def test_invalid_final_citation_is_removed(provider, evidence):
    invalid = {**ANSWER, 'claims': [{**ANSWER['claims'][0], 'citation_ids': ['invented']}]}
    provider([{'answer': ANSWER, 'decision': DONE},
              {'critique': {'verdicts': [{**AUDIT['verdicts'][0], 'status': 'unsupported'}]}, 'final': invalid}])
    assert pipeline.run_pipeline(AskRequest(question='How does it work?'), Retriever(evidence)).final.abstained


def test_whole_chunks_are_preserved_under_budget(evidence):
    from reasoning.efficient import select_evidence
    long = evidence.model_copy(update={'chunk_id': 'large', 'text': 'x' * 8001})
    fitting = evidence.model_copy(update={'chunk_id': 'fit', 'text': 'y' * 7990})
    assert select_evidence([long, fitting, evidence]) == [fitting]


def test_json_repair_counts_toward_usage(provider):
    model = provider(['invalid', {'subquestions': ['first', 'second']}])
    with track_usage('efficient') as usage:
        assert json_call('Decompose', Decomposition).subquestions == ['first', 'second']
    assert usage.report()['model_calls'] == 2 and usage.report()['prompt_tokens'] == 200
    assert 'your last response was not valid JSON — return only the JSON object' in model.chat.call_args.kwargs['messages'][-1]['content']


def test_call_and_context_limits_stop_before_provider(provider):
    model = provider(['{}'] * 6)
    with track_usage('efficient') as usage:
        with pytest.raises(client.BudgetExceeded):
            client.llm_call('x' * 32001)
        assert model.chat.call_count == 0
        for _ in range(6):
            client.llm_call('{}')
        with pytest.raises(client.BudgetExceeded):
            client.llm_call('{}')
    assert model.chat.call_count == usage.report()['model_calls'] == 6


def test_quota_latch_blocks_subsequent_requests_and_records_failure(monkeypatch, tmp_path):
    model = Mock()
    model.chat.side_effect = RuntimeError('token_quota_reached secret-account-identifier')
    monkeypatch.setattr(client, '_model', lambda: model)
    with track_usage('efficient') as usage:
        for _ in range(2):
            with pytest.raises(client.QuotaExceeded) as error:
                client.llm_call('secret question')
            assert 'secret' not in str(error.value)
    assert model.chat.call_count == 1 and usage.report()['prompt_tokens'] is None
    ledger = (tmp_path / 'usage.jsonl').read_text()
    assert 'secret' not in ledger and 'quota_exhausted' in ledger


def test_missing_usage_is_unknown_not_zero(monkeypatch):
    model = Mock()
    model.chat.return_value = {'choices': [{'message': {'content': '{}'}}]}
    monkeypatch.setattr(client, '_model', lambda: model)
    with track_usage('efficient') as usage:
        client.llm_call('test')
    assert usage.report()['prompt_tokens'] is None and not usage.report()['usage_complete']


def test_combined_review_preserves_supported_claims_and_rejects_additions(provider, evidence):
    changed = {**ANSWER, 'claims': [{**ANSWER['claims'][0], 'text': 'An unrelated assertion.'},
                                  {'claim_id': 'new', 'text': 'Another assertion.', 'citation_ids': ['c1']}]}
    provider([{'answer': ANSWER, 'decision': DONE}, {'critique': AUDIT, 'final': changed}])
    result = pipeline.run_pipeline(AskRequest(question='How does it work?'), Retriever(evidence))
    assert result.final.claims == result.draft.claims


def test_benchmark_stops_at_first_quota_error(monkeypatch, tmp_path):
    import sys
    from eval import run_eval
    import retrieval.retriever
    questions = [{'id': str(i), 'question': 'What evidence exists?', 'expected_papers': [],
                  'kind': 'single_hop', 'should_abstain': False} for i in range(18)]
    question_path = tmp_path / 'questions.json'
    question_path.write_text(json.dumps(questions))
    retriever = Mock()
    retriever.collection.get.return_value = {'metadatas': []}
    monkeypatch.setattr(retrieval.retriever, 'Retriever', lambda: retriever)
    runner = Mock(side_effect=client.QuotaExceeded('Quota exhausted'))
    monkeypatch.setattr(run_eval, 'run_pipeline', runner)
    monkeypatch.setattr(sys, 'argv', ['eval', '--questions', str(question_path), '--output', str(tmp_path / 'results')])
    with pytest.raises(SystemExit, match='incomplete'):
        run_eval.main()
    assert runner.call_count == 1
    assert runner.call_args.args[0].mode == 'baseline' and not runner.call_args.args[0].use_cache
    saved = json.loads(next((tmp_path / 'results').glob('*/run.json')).read_text())
    assert len(saved['not_attempted']) == 17 and len(saved['failures']) == 1
    assert not saved['completed']


def test_acceptance_stops_before_remaining_probes(monkeypatch, tmp_path):
    from eval import run_acceptance
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_acceptance, 'Retriever', Mock())
    runner = Mock(side_effect=client.QuotaExceeded('Quota exhausted'))
    monkeypatch.setattr(run_acceptance, 'run_pipeline', runner)
    with pytest.raises(SystemExit, match='exhausted quota'):
        run_acceptance.main()
    assert runner.call_count == 1
    saved = json.loads(next(tmp_path.glob('eval/results/*/acceptance.json')).read_text())
    assert saved['checks']['no_answer']['not_attempted']
    assert saved['checks']['controlled_revision']['not_attempted']
