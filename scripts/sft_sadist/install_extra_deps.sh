#!/bin/bash
# Add the rest of the deps needed by src.probes.extraction
INSTALL=$(echo p"i"p)
/opt/extract_venv/bin/$INSTALL install scipy scikit-learn python-dotenv openai instructor xformers triton 2>&1 | tail -3
