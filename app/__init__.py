"""EMVA Studio: the hosted internal tool (Phase 8). Streamlit UI over the ``emva`` package.

Pure modules (no Streamlit import): ``storage`` (dataset store, run registry), ``validation`` (upload
schema checks), ``training`` (subprocess training, pre-training summary), ``job`` (the training job runner),
``results`` (evaluation frames for a run), ``scoring`` (form adapter over ``emva.scoring``) and ``auth``
(password check). UI modules: ``main`` (entrypoint, gate, navigation), ``theme``, ``charts``, ``components``
and the page scripts in ``views/``.
"""
