"""Collection controls and claim-to-source exploration."""
import requests
import streamlit as st


def collection_picker(api_url):
    try:
        response = requests.get(api_url + '/collections', timeout=(3, 10))
        response.raise_for_status()
        collections = response.json()
    except (requests.RequestException, ValueError):
        st.caption('Collection manager unavailable. Start the research API to manage papers.')
        collections = [{'id': 'default', 'name': 'RAG & multi-hop QA', 'read_only': True}]
    ids = [c['id'] for c in collections]
    names = {c['id']: c['name'] for c in collections}
    pending = st.session_state.pop('new_collection_id', None)
    if pending in ids:
        st.session_state.collection_picker = pending
    if st.session_state.get('collection_picker') not in ids:
        st.session_state.collection_picker = ids[0]
    cid = st.selectbox('Research collection', ids, format_func=lambda x: names[x], key='collection_picker')
    if st.session_state.get('previous_collection', 'default') != cid:
        st.session_state.pop('result', None)
    st.session_state.previous_collection = cid
    selected = next(c for c in collections if c['id'] == cid)
    with st.expander('Manage papers'):
        with st.form('create_collection'):
            name = st.text_input('New collection name', max_chars=80, placeholder='e.g. Climate adaptation')
            if st.form_submit_button('Create collection'):
                if not name.strip():
                    st.error('Enter a collection name.')
                else:
                    try:
                        response = requests.post(api_url + '/collections', json={'name': name}, timeout=(5, 30))
                        response.raise_for_status()
                        st.session_state.new_collection_id = response.json()['id']
                        st.rerun()
                    except requests.RequestException:
                        st.error('Could not create the collection. Check the research API.')
        if selected['read_only']:
            st.caption('The starter corpus is read-only here. Create a collection to add your own papers.')
        else:
            with st.form('upload_papers', clear_on_submit=True):
                files = st.file_uploader('Add PDFs', type=['pdf'], accept_multiple_files=True)
                st.caption('Up to 20 MB and 200 pages per PDF. Text-based PDFs only; no OCR. Files stay local. Asking a question sends retrieved text to watsonx.')
                upload = st.form_submit_button('Index PDFs')
            if upload:
                if not files:
                    st.info('Choose at least one PDF.')
                for file in files:
                    if file.size > 20 * 1024 * 1024:
                        st.error(f'{file.name}: exceeds 20 MB.')
                        continue
                    try:
                        with st.spinner(f'Indexing {file.name} locally…'):
                            response = requests.post(api_url + f'/collections/{cid}/papers',
                                params={'filename': file.name}, data=file.getvalue(),
                                headers={'Content-Type': 'application/pdf'}, timeout=(10, 600))
                        if response.ok:
                            doc = response.json()
                            st.success(f"{doc['title']}: {'already indexed' if doc['already_indexed'] else str(doc['chunks']) + ' passages ready'}")
                        else:
                            st.error(response.json().get('detail', 'Upload failed.'))
                    except (requests.RequestException, ValueError):
                        st.error(f'{file.name}: indexing could not be confirmed. Retrying the same PDF will not duplicate it.')
        try:
            response = requests.get(api_url + f'/collections/{cid}/papers', timeout=(3, 10))
            response.raise_for_status()
            docs = response.json()
            noun = 'paper' if len(docs) == 1 else 'papers'
            st.caption(f'{len(docs)} {noun} in this collection')
            for doc in docs:
                st.text(doc['title'])
            if not docs:
                st.info('Add PDFs before asking a question.')
        except (requests.RequestException, ValueError):
            st.caption('Paper list is currently unavailable.')
    return cid, names[cid]


def source_url(url, api_url):
    if url.startswith(('http://', 'https://')):
        return url
    if url.startswith('/collections/'):
        return api_url + url
    return None


def evidence_card(item, cid, api_url, key):
    st.markdown(f"**{item['title']}**")
    st.caption(f"{item['section']} · PDF pages {item['page_start']}–{item['page_end']}")
    st.caption('Exact cited excerpt supplied to the model')
    st.text(item['text'])
    link = source_url(item['url'], api_url)
    if link:
        st.link_button('Open paper', link)
    page = st.number_input('PDF page to inspect', min_value=1, value=item['page_start'], step=1, key=key+'_page')
    if st.button('Show source page', key=key+'_show'):
        endpoint = api_url + f"/collections/{cid}/papers/{item['paper_id']}/pages/{page}"
        try:
            response = requests.get(endpoint, timeout=(5, 60))
            response.raise_for_status()
            picture = requests.get(endpoint + '/image', timeout=(5, 60))
            picture.raise_for_status()
            st.session_state[key+'_context'] = {'page': page, 'text': response.json()['text'], 'image': picture.content}
        except (requests.RequestException, ValueError):
            st.session_state.pop(key+'_context', None)
            st.warning('This source page is unavailable. Check the page number and that the original PDF is still in this collection.')
    context = st.session_state.get(key+'_context')
    if context and context['page'] == page:
        st.info('Source-page context for your inspection. Only the exact excerpt above is guaranteed to have been supplied with this citation; additional page text does not retroactively support the saved answer.')
        st.image(context['image'], caption=f"Original PDF · page {page}")
        with st.expander('Page text'):
            st.text(context['text'])
    st.caption(f"Evidence ID: {item['chunk_id']}")
