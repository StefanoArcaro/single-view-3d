.PHONY: install run start clean

# Install your project in editable mode
install:
	uv pip install -e .  # or pip install -e .

# Run the Streamlit app from repo root
run:
	streamlit run streamlit/Homepage.py

# Clean pycache and build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf *.egg-info

# Clean, install, and run everything
start:
	$(MAKE) clean
	$(MAKE) install
	$(MAKE) run