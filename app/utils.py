import os.path as op

import streamlit as st


BASE_DIR = op.dirname(__file__)
STATIC_DIR = op.join(BASE_DIR, "static")


def st_markdown_file(filename: str, *args, **kwargs):
    with open(op.join(STATIC_DIR, filename)) as f:
        s = f.read()
    return st.markdown(s, *args, **kwargs)
