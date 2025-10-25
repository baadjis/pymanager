#!/bin/bash
# start.sh - PyManager Smart Startup Script
# Usage: ./start.sh [dev|test|prod]

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

ENVIRONMENT="${1:-dev}"  # dev par défaut
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON="$VENV_DIR/bin/python"
STREAMLIT="$VENV_DIR/bin/streamlit"
MCP_SERVER_PORT=8000
STREAMLIT_PORT=8501

# PID files
MCP_PID_FILE="$PROJECT_DIR/.mcp_server.pid"
STREAMLIT_PID_FILE="$PROJECT_DIR/.streamlit.pid"
MONGODB_PID_FILE="/var/run/mongodb/mongod.pid"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Functions
# ============================================================================

print_info() {
    echo -e "${BLUE}ℹ ${1}${NC}"
}

print_success() {
    echo -e "${GREEN}✓ ${1}${NC}"
}

print_error() {
    echo -e "${RED}✗ ${1}${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ ${1}${NC}"
}

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}${1}${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if port is in use
port_in_use() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

# Kill process on port
kill_port() {
    local port=$1
    if port_in_use $port; then
        print_warning "Killing process on port $port..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

# Check MongoDB status
check_mongodb() {
    if command_exists mongosh; then
        mongosh --quiet --eval "db.adminCommand('ping')" >/dev/null 2>&1
    elif command_exists mongo; then
        mongo --quiet --eval "db.adminCommand('ping')" >/dev/null 2>&1
    else
        return 1
    fi
}

# Start MongoDB
start_mongodb() {
    print_info "Checking MongoDB..."
    
    if check_mongodb; then
        print_success "MongoDB already running"
        return 0
    fi
    
    print_info "Starting MongoDB..."
    
    # Try different methods based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command_exists brew; then
            brew services start mongodb-community 2>/dev/null || \
            brew services start mongodb/brew/mongodb-community 2>/dev/null || {
                print_error "Failed to start MongoDB via brew"
                print_warning "Try: brew services start mongodb-community@7.0"
                return 1
            }
        else
            mongod --config /usr/local/etc/mongod.conf --fork 2>/dev/null || {
                print_error "Failed to start MongoDB"
                return 1
            }
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command_exists systemctl; then
            sudo systemctl start mongodb 2>/dev/null || \
            sudo systemctl start mongod 2>/dev/null || {
                print_error "Failed to start MongoDB via systemctl"
                return 1
            }
        else
            sudo service mongodb start 2>/dev/null || \
            sudo service mongod start 2>/dev/null || {
                print_error "Failed to start MongoDB via service"
                return 1
            }
        fi
    fi
    
    # Wait for MongoDB to be ready
    print_info "Waiting for MongoDB to be ready..."
    for i in {1..30}; do
        if check_mongodb; then
            print_success "MongoDB started successfully"
            return 0
        fi
        sleep 1
    done
    
    print_error "MongoDB failed to start after 30 seconds"
    return 1
}

# Start MCP Server
start_mcp_server() {
    print_info "Starting MCP Server..."
    
    # Check if already running
    if [ -f "$MCP_PID_FILE" ] && kill -0 $(cat "$MCP_PID_FILE") 2>/dev/null; then
        print_success "MCP Server already running (PID: $(cat "$MCP_PID_FILE"))"
        return 0
    fi
    
    # Kill if port in use
    kill_port $MCP_SERVER_PORT
    
    # Set environment
    export ENVIRONMENT=$ENVIRONMENT
    export MONGODB_URI="${MONGODB_URI:-mongodb://localhost:27017/}"
    
    # Start server
    cd "$PROJECT_DIR"
    nohup "$PYTHON" mcp_server.py > logs/mcp_server.log 2>&1 &
    echo $! > "$MCP_PID_FILE"
    
    # Wait for server to be ready
    print_info "Waiting for MCP Server..."
    for i in {1..15}; do
        if curl -s http://localhost:$MCP_SERVER_PORT/ >/dev/null 2>&1; then
            print_success "MCP Server started (PID: $(cat "$MCP_PID_FILE"))"
            return 0
        fi
        sleep 1
    done
    
    print_error "MCP Server failed to start"
    cat logs/mcp_server.log
    return 1
}

# Start Streamlit
start_streamlit() {
    print_info "Starting Streamlit..."
    
    # Check if already running
    if [ -f "$STREAMLIT_PID_FILE" ] && kill -0 $(cat "$STREAMLIT_PID_FILE") 2>/dev/null; then
        print_success "Streamlit already running (PID: $(cat "$STREAMLIT_PID_FILE"))"
        return 0
    fi
    
    # Kill if port in use
    kill_port $STREAMLIT_PORT
    
    # Set environment
    export ENVIRONMENT=$ENVIRONMENT
    
    # Streamlit config based on environment
    if [ "$ENVIRONMENT" = "prod" ]; then
        HEADLESS="true"
        GATHER_STATS="false"
    else
        HEADLESS="true"
        GATHER_STATS="false"
    fi
    
    # Start Streamlit
    cd "$PROJECT_DIR"
    nohup "$STREAMLIT" run app3.py \
        --server.port=$STREAMLIT_PORT \
        --server.headless=$HEADLESS \
        --browser.gatherUsageStats=$GATHER_STATS \
        --server.enableCORS=false \
        --server.enableXsrfProtection=true \
        > logs/streamlit.log 2>&1 &
    
    echo $! > "$STREAMLIT_PID_FILE"
    
    # Wait for Streamlit
    print_info "Waiting for Streamlit..."
    for i in {1..20}; do
        if curl -s http://localhost:$STREAMLIT_PORT/_stcore/health >/dev/null 2>&1; then
            print_success "Streamlit started (PID: $(cat "$STREAMLIT_PID_FILE"))"
            return 0
        fi
        sleep 1
    done
    
    print_error "Streamlit failed to start"
    cat logs/streamlit.log
    return 1
}

# Stop all services
stop_services() {
    print_info "Stopping services..."
    
    # Stop Streamlit
    if [ -f "$STREAMLIT_PID_FILE" ]; then
        PID=$(cat "$STREAMLIT_PID_FILE")
        if kill -0 $PID 2>/dev/null; then
            kill $PID 2>/dev/null || kill -9 $PID 2>/dev/null
            print_success "Streamlit stopped"
        fi
        rm -f "$STREAMLIT_PID_FILE"
    fi
    
    # Stop MCP Server
    if [ -f "$MCP_PID_FILE" ]; then
        PID=$(cat "$MCP_PID_FILE")
        if kill -0 $PID 2>/dev/null; then
            kill $PID 2>/dev/null || kill -9 $PID 2>/dev/null
            print_success "MCP Server stopped"
        fi
        rm -f "$MCP_PID_FILE"
    fi
}

# Cleanup on exit
cleanup() {
    print_info "Cleaning up..."
    stop_services
}

# Status check
check_status() {
    print_header "Service Status"
    
    # MongoDB
    if check_mongodb; then
        print_success "MongoDB: Running"
    else
        print_error "MongoDB: Stopped"
    fi
    
    # MCP Server
    if [ -f "$MCP_PID_FILE" ] && kill -0 $(cat "$MCP_PID_FILE") 2>/dev/null; then
        print_success "MCP Server: Running (PID: $(cat "$MCP_PID_FILE"))"
    else
        print_error "MCP Server: Stopped"
    fi
    
    # Streamlit
    if [ -f "$STREAMLIT_PID_FILE" ] && kill -0 $(cat "$STREAMLIT_PID_FILE") 2>/dev/null; then
        print_success "Streamlit: Running (PID: $(cat "$STREAMLIT_PID_FILE"))"
    else
        print_error "Streamlit: Stopped"
    fi
}

# ============================================================================
# Main
# ============================================================================

main() {
    clear
    print_header "🚀 PyManager Startup - $ENVIRONMENT"
    
    # Validate environment
    if [[ ! "$ENVIRONMENT" =~ ^(dev|test|prod)$ ]]; then
        print_error "Invalid environment: $ENVIRONMENT"
        echo "Usage: $0 [dev|test|prod]"
        exit 1
    fi
    
    # Check prerequisites
    print_info "Checking prerequisites..."
    
    if [ ! -d "$VENV_DIR" ]; then
        print_error "Virtual environment not found: $VENV_DIR"
        print_warning "Run: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
        exit 1
    fi
    
    if [ ! -f "$PYTHON" ]; then
        print_error "Python not found in venv"
        exit 1
    fi
    
    if [ ! -f "$STREAMLIT" ]; then
        print_error "Streamlit not found in venv"
        print_warning "Run: venv/bin/pip install streamlit"
        exit 1
    fi
    
    if ! command_exists mongod && ! command_exists mongo; then
        print_error "MongoDB not installed"
        print_warning "Install: brew install mongodb-community (macOS) or apt-get install mongodb (Linux)"
        exit 1
    fi
    
    print_success "Prerequisites OK"
    
    # Create logs directory
    mkdir -p logs
    
    # Setup trap for cleanup
    trap cleanup EXIT INT TERM
    
    # Start services
    echo ""
    print_header "Starting Services"
    
    # 1. MongoDB
    if ! start_mongodb; then
        print_error "Failed to start MongoDB"
        exit 1
    fi
    
    # 2. MCP Server
    if ! start_mcp_server; then
        print_error "Failed to start MCP Server"
        exit 1
    fi
    
    # 3. Streamlit
    if ! start_streamlit; then
        print_error "Failed to start Streamlit"
        exit 1
    fi
    
    # Summary
    echo ""
    print_header "✅ PyManager Started Successfully"
    echo ""
    print_success "Environment: $ENVIRONMENT"
    print_success "MongoDB: Running"
    print_success "MCP Server: http://localhost:$MCP_SERVER_PORT"
    print_success "Streamlit: http://localhost:$STREAMLIT_PORT"
    echo ""
    print_info "Logs:"
    echo "  - MCP Server: logs/mcp_server.log"
    echo "  - Streamlit: logs/streamlit.log"
    echo ""
    print_warning "Press Ctrl+C to stop all services"
    echo ""
    
    # Monitor services
    while true; do
        sleep 5
        
        # Check if services are still running
        if [ -f "$MCP_PID_FILE" ] && ! kill -0 $(cat "$MCP_PID_FILE") 2>/dev/null; then
            print_error "MCP Server crashed! Check logs/mcp_server.log"
            exit 1
        fi
        
        if [ -f "$STREAMLIT_PID_FILE" ] && ! kill -0 $(cat "$STREAMLIT_PID_FILE") 2>/dev/null; then
            print_error "Streamlit crashed! Check logs/streamlit.log"
            exit 1
        fi
    done
}

# ============================================================================
# CLI
# ============================================================================

case "${2:-}" in
    status)
        check_status
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 2
        main
        ;;
    logs)
        tail -f logs/mcp_server.log logs/streamlit.log
        ;;
    *)
        main
        ;;
esac
