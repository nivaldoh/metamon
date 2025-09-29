# Create a notebook instance with GPUs
# Choose a machine type with appropriate GPUs (e.g., n1-standard-8 with NVIDIA T4/V100/A100)

# Once in the notebook, clone and install Metamon
git clone --recursive https://github.com/UT-Austin-RPL/metamon.git
cd metamon
pip install -e .

# Install AMAGO (the underlying framework)
cd amago
pip install -e amago

# Set up Pokemon Showdown server
cd ../server/pokemon-showdown
npm install
node pokemon-showdown start --no-security &

# Use Google Cloud Storage for datasets
export METAMON_CACHE_DIR=/cache
# Or mount a GCS bucket using gcsfuse