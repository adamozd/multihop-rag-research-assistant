"""Live acceptance probes, separate from the frozen benchmark and human scores."""
import json
from datetime import datetime, timezone
from pathlib import Path

from llm.watsonx_client import QuotaExceeded
from llm.usage import track_usage
from reasoning.critic import critique
from reasoning.models import Answer, AskRequest, Evidence
from reasoning.pipeline import run_pipeline
from reasoning.synthesizer import synthesize
from retrieval.retriever import Retriever


def main():
    output = Path('eval/results') / ('acceptance-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True)
    retriever = Retriever()
    report = {'measurement': 'live_acceptance_probes_not_human_review', 'mode': 'baseline', 'checks': {},
              'manual_review_required': [
                  'Do decomposition and follow-up searches stay on retrieval timing?',
                  'Does the answer correctly distinguish the optional threshold policy?',
                  'Do the cited passages entail every final claim?',
                  'Does revision correct/remove the error everywhere it recurs?']}

    def save():
        (output / 'acceptance.json').write_text(json.dumps(report, indent=2))

    probes = {
        'comparison': 'How do IRCoT and Self-RAG decide when to retrieve evidence, and how do their approaches differ?',
        'no_answer': "Which theorem in these papers proves a single self-critique revision eliminates every unsupported research claim?",
    }
    for name, question in probes.items():
        try:
            result = run_pipeline(AskRequest(mode="baseline", use_cache=False, question=question), retriever)
            (output / f'{name}.json').write_text(result.model_dump_json(indent=2))
            report['checks'][name] = {'executed': True, 'abstained': result.final.abstained,
                                     'revised': result.revised, 'warnings': result.warnings}
            if name == 'no_answer':
                report['checks'][name]['abstention_pass'] = result.final.abstained
        except Exception as exc:
            report['checks'][name] = {'executed': False, 'error': str(exc)}
            if isinstance(exc, QuotaExceeded):
                for pending in (*probes, 'controlled_revision'):
                    report['checks'].setdefault(pending, {'executed': False, 'not_attempted': True,
                                                          'reason': 'token_quota_exhausted'})
                save()
                raise SystemExit(f'Acceptance stopped at exhausted quota. Results: {output}') from exc
        save()
        print(f'Finished {name}', flush=True)

    # Controlled diagnostic: supply the actual primary-source inference passage,
    # so this checks the critic separately from retrieval success. This seeded
    # error is never counted as a naturally occurring benchmark revision.
    try:
        with track_usage('baseline') as usage:
            raw = retriever.collection.get(where={'paper_id': '2310.11511'}, include=['documents', 'metadatas'])
            evidence = [Evidence(chunk_id=cid, text=text, **metadata)
                        for cid, text, metadata in zip(raw['ids'], raw['documents'], raw['metadatas'])
                        if '3.3' in metadata['section'] and 'threshold' in text.lower() and 'alternatively' in text.lower()]
            if not evidence:
                raise ValueError('Primary Self-RAG inference passage unavailable; controlled probe not run.')
            evidence = evidence[:1]
            draft = Answer.model_validate({'claims': [{'claim_id': 'seeded_error',
                'text': 'Self-RAG always requires both a retrieval-token decision and a probability threshold to trigger retrieval at inference.',
                'citation_ids': [evidence[0].chunk_id]}], 'abstained': False, 'limitations': []})
            first = critique(draft, evidence)
            flagged = any(v.status != 'supported' for v in first.verdicts)
            final = synthesize('How does Self-RAG decide when to retrieve?', evidence, draft, first) if flagged else draft
            second = critique(final, evidence) if flagged else first
            record = {'kind': 'controlled_seeded_error_not_benchmark', 'evidence': [e.model_dump() for e in evidence],
                      'draft': draft.model_dump(), 'first_audit': first.model_dump(),
                      'final': final.model_dump(), 'second_audit': second.model_dump()}
            (output / 'controlled-revision.json').write_text(json.dumps(record, indent=2))
            report['checks']['controlled_revision'] = {'executed': True, 'seeded_error_flagged': flagged,
                'revision_ran': flagged, 'final_internal_flags': sum(v.status != 'supported' for v in second.verdicts),
                'human_semantic_verdict': None}
            report['checks']['controlled_revision']['usage'] = usage.report()
    except Exception as exc:
        report['checks']['controlled_revision'] = {'executed': False, 'error': str(exc)}
    save()
    print(f'Results: {output}', flush=True)
    if any(not item['executed'] for item in report['checks'].values()):
        raise SystemExit('Some acceptance probes did not execute; inspect acceptance.json.')


if __name__ == '__main__':
    main()
