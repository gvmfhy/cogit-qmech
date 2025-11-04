#!/bin/bash
# sync_to_remote.sh - Sync cogit-qmech project to/from remote GPU instances
#
# Usage:
#   ./sync_to_remote.sh push digitalocean <IP>
#   ./sync_to_remote.sh push runpod <IP> <PORT>
#   ./sync_to_remote.sh pull digitalocean <IP>
#   ./sync_to_remote.sh pull runpod <IP> <PORT>

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Parse arguments
ACTION=$1    # push or pull
PROVIDER=$2  # digitalocean or runpod
HOST=$3      # IP address
PORT=${4:-22}  # Port (default 22)

# Validation
if [ -z "$ACTION" ] || [ -z "$PROVIDER" ] || [ -z "$HOST" ]; then
    echo -e "${RED}Usage:${NC}"
    echo "  Push to remote:"
    echo "    $0 push digitalocean <IP>"
    echo "    $0 push runpod <IP> <PORT>"
    echo "  Pull from remote:"
    echo "    $0 pull digitalocean <IP>"
    echo "    $0 pull runpod <IP> <PORT>"
    exit 1
fi

# Directories to sync
SRC_DIR="$SCRIPT_DIR/"
DEST_DIR="root@$HOST:~/cogit-qmech/"

# Exclude patterns
EXCLUDES=(
    --exclude '.git'
    --exclude '__pycache__'
    --exclude '*.pyc'
    --exclude '.venv'
    --exclude 'venv'
    --exclude '*.log'
    --exclude '.DS_Store'
    --exclude 'phase*_test_output.log'
)

# Build rsync command based on provider
if [ "$PROVIDER" = "digitalocean" ]; then
    RSYNC_CMD="rsync -avz --progress ${EXCLUDES[@]}"
elif [ "$PROVIDER" = "runpod" ]; then
    RSYNC_CMD="rsync -avz --progress -e 'ssh -p $PORT' ${EXCLUDES[@]}"
else
    echo -e "${RED}Error: Unknown provider '$PROVIDER'. Use 'digitalocean' or 'runpod'.${NC}"
    exit 1
fi

# Execute based on action
if [ "$ACTION" = "push" ]; then
    echo -e "${BLUE}Pushing project to $PROVIDER ($HOST:$PORT)...${NC}"
    $RSYNC_CMD "$SRC_DIR" "$DEST_DIR"
    echo -e "${GREEN}✓ Push complete!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps on remote:${NC}"
    echo "  ssh root@$HOST $([ "$PORT" != "22" ] && echo "-p $PORT")"
    echo "  cd ~/cogit-qmech"
    echo "  source .venv/bin/activate"
    echo "  python experiments/sentiment/quantum_phase1_collect.py --preset qwen_remote"

elif [ "$ACTION" = "pull" ]; then
    echo -e "${BLUE}Pulling results from $PROVIDER ($HOST:$PORT)...${NC}"
    
    # Pull specific directories
    echo -e "${YELLOW}Pulling results/...${NC}"
    $RSYNC_CMD "root@$HOST:~/cogit-qmech/results/" "$SRC_DIR/results/" || echo "No results/ directory"
    
    echo -e "${YELLOW}Pulling models/...${NC}"
    $RSYNC_CMD "root@$HOST:~/cogit-qmech/models/" "$SRC_DIR/models/" || echo "No models/ directory"
    
    echo -e "${YELLOW}Pulling data/...${NC}"
    $RSYNC_CMD "root@$HOST:~/cogit-qmech/data/" "$SRC_DIR/data/" || echo "No data/ directory"
    
    echo -e "${GREEN}✓ Pull complete!${NC}"
    echo ""
    echo -e "${YELLOW}Check these directories for new files:${NC}"
    echo "  results/"
    echo "  models/"
    echo "  data/"

else
    echo -e "${RED}Error: Unknown action '$ACTION'. Use 'push' or 'pull'.${NC}"
    exit 1
fi
