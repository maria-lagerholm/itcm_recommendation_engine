#!/bin/bash

# Setup script for ITCM Recommendation Engine environment variables

echo "Setting up environment variables for ITCM Recommendation Engine..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cat > .env << EOF
# Bring/Mybring API Credentials
# These are used for postal code lookups in the customer data analysis
MYBRING_UID=maria.lagerholm@itcm.se
MYBRING_KEY=80c60aee-ceb7-4e47-9ab5-ec30a7da4e9d
EOF
    echo "Created .env file with Bring/Mybring API credentials"
else
    echo ".env file already exists"
fi

# Export environment variables for current session
export MYBRING_UID="maria.lagerholm@itcm.se"
export MYBRING_KEY="80c60aee-ceb7-4e47-9ab5-ec30a7da4e9d"

echo "Environment variables set for current session:"
echo "MYBRING_UID: $MYBRING_UID"
echo "MYBRING_KEY: $MYBRING_KEY"

echo ""
echo "To make these environment variables permanent, add the following to your ~/.bashrc or ~/.zshrc:"
echo "export MYBRING_UID=\"maria.lagerholm@itcm.se\""
echo "export MYBRING_KEY=\"80c60aee-ceb7-4e47-9ab5-ec30a7da4e9d\""
echo ""
echo "Or source the .env file in your shell configuration:"
echo "source .env" 