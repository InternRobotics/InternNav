#!/bin/bash
# Start Agent Server for InternVLA-N1 Testing
#
# Usage:
#   ./scripts/start_test_server.sh [port]
#
# Default port: 8087

PORT=${1:-8087}

echo "=========================================="
echo "Starting InternNav Agent Server"
echo "=========================================="
echo "Port: $PORT"
echo "=========================================="
echo ""
echo "The server will start and wait for agent initialization."
echo "After the server starts, run the test script in another terminal:"
echo ""
echo "  python scripts/test_internvla_n1.py --checkpoint <checkpoint_path>"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""
echo "=========================================="
echo ""

python scripts/eval/start_server.py --port $PORT
