#!/bin/bash
# Multi-Turn Continuation Environment Setup Script

echo "========================================="
echo "Multi-Turn Continuation - Environment Setup"
echo "========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}✗ pip3 not found. Please install pip.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ pip3 found${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}✗ Error: requirements.txt not found.${NC}"
    echo "  Please run this script from the CPB-Bench directory."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
echo "  This may take a few minutes..."
pip3 install -q -r requirements.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi
echo ""

# Check for API keys
echo "Checking API keys..."
api_keys_found=0

if [ -n "$OPENAI_API_KEY" ]; then
    echo -e "${GREEN}✓ OPENAI_API_KEY found${NC}"
    api_keys_found=$((api_keys_found + 1))
else
    echo -e "${YELLOW}⚠ OPENAI_API_KEY not set${NC}"
fi

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo -e "${GREEN}✓ ANTHROPIC_API_KEY found${NC}"
    api_keys_found=$((api_keys_found + 1))
else
    echo -e "${YELLOW}⚠ ANTHROPIC_API_KEY not set (optional)${NC}"
fi

if [ -n "$GEMINI_API_KEY" ]; then
    echo -e "${GREEN}✓ GEMINI_API_KEY found${NC}"
    api_keys_found=$((api_keys_found + 1))
else
    echo -e "${YELLOW}⚠ GEMINI_API_KEY not set (optional)${NC}"
fi

echo ""

if [ $api_keys_found -eq 0 ]; then
    echo -e "${YELLOW}=========================================${NC}"
    echo -e "${YELLOW}No API keys found!${NC}"
    echo -e "${YELLOW}=========================================${NC}"
    echo ""
    echo "To use OpenAI models (gpt-4o-mini, gpt-4, gpt-5), you need to set your API key:"
    echo ""
    echo "  export OPENAI_API_KEY='sk-...'"
    echo ""
    echo "Or create a .env file in the project root directory:"
    echo ""
    echo "  cp .env.example .env"
    echo "  # Edit .env and add your API keys"
    echo "  nano .env"
    echo ""
    echo "Then reload your environment:"
    echo ""
    echo "  export \$(cat .env | grep -v '^#' | xargs)"
    echo ""
else
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}Setup Complete!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo ""
    echo "You can now run the multi-turn continuation:"
    echo ""
    echo "1. Prepare data (if not already done):"
    echo "   python multiturn_continuation/scripts/prepare_data.py \\"
    echo "       --excel data/Finalized_detailed_results_by_category.xlsx \\"
    echo "       --source_data data_loader/output_benchmark \\"
    echo "       --output multiturn_continuation/data_processing/output/failed_cases_multiturn.json"
    echo ""
    echo "2. Run continuation (test with 2 cases):"
    echo "   python multiturn_continuation/scripts/run_continuation.py \\"
    echo "       --input multiturn_continuation/data_processing/output/failed_cases_multiturn.json \\"
    echo "       --doctor_model gpt-4o-mini \\"
    echo "       --max_turns 3 \\"
    echo "       --sample 2 \\"
    echo "       --output multiturn_continuation/output/test_results.json"
    echo ""
fi

echo "For more information, see:"
echo "  - multiturn_continuation/README.md"
echo ""
