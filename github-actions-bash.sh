#!/bin/bash
# GitHub Actions Monitor for Bash Terminal
# Simple version that works without sed, grep, awk

OWNER="kalidasan-2001"
REPO="qlorax-updated"
API_BASE="https://api.github.com"

# Colors (if terminal supports them)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    local color=$1
    local text=$2
    echo -e "${color}${text}${NC}"
}

# Check if GitHub CLI is authenticated
check_auth() {
    if ! gh auth status &>/dev/null; then
        print_color "$RED" "❌ GitHub CLI not authenticated!"
        echo "Please run: gh auth login"
        exit 1
    fi
}

# Function to call GitHub API using gh CLI
call_github_api() {
    local endpoint="$1"
    gh api "/repos/${OWNER}/${REPO}/${endpoint}"
}

# Show latest workflow status
show_status() {
    print_color "$BLUE" "📋 Latest GitHub Actions Status"
    echo "================================================================"
    echo "Repository: $OWNER/$REPO"
    echo "Time: $(date)"
    echo ""
    
    # Get latest workflow run
    local response=$(call_github_api "actions/runs?per_page=1")
    
    if [ $? -ne 0 ] || [ -z "$response" ]; then
        print_color "$RED" "❌ Failed to fetch workflow data"
        echo "Check your GitHub token and internet connection"
        return 1
    fi
    
    # Simple parsing without sed/grep
    echo "Raw API Response (first 1000 characters):"
    echo "$response" | head -c 1000
    echo ""
    echo "..."
    echo ""
    
    print_color "$CYAN" "🔗 View detailed logs at:"
    echo "https://github.com/$OWNER/$REPO/actions"
}

# List recent workflow runs
list_runs() {
    print_color "$BLUE" "🔄 Recent GitHub Actions Runs"
    echo "================================================================"
    
    local response=$(call_github_api "actions/runs?per_page=5")
    
    if [ $? -ne 0 ] || [ -z "$response" ]; then
        print_color "$RED" "❌ Failed to fetch workflow runs"
        return 1
    fi
    
    echo "API Response:"
    echo "$response"
    echo ""
    
    print_color "$CYAN" "🔗 View all runs at:"
    echo "https://github.com/$OWNER/$REPO/actions"
}

# Show failed runs
show_failed() {
    print_color "$RED" "❌ Checking for Failed Runs"
    echo "================================================================"
    
    local response=$(call_github_api "actions/runs?status=failure&per_page=5")
    
    if [ $? -ne 0 ] || [ -z "$response" ]; then
        print_color "$RED" "❌ Failed to fetch failed runs"
        return 1
    fi
    
    echo "Failed Runs Response:"
    echo "$response"
    echo ""
    
    print_color "$CYAN" "🔗 View failed runs at:"
    echo "https://github.com/$OWNER/$REPO/actions?query=is%3Afailure"
}

# Monitor in real-time
monitor() {
    local interval=${1:-30}
    print_color "$BLUE" "🔄 Monitoring GitHub Actions (refresh every ${interval}s)"
    print_color "$YELLOW" "Press Ctrl+C to stop"
    echo "================================================================"
    
    while true; do
        clear
        show_status
        echo ""
        print_color "$CYAN" "Next refresh in ${interval} seconds..."
        echo "================================================================"
        sleep "$interval"
    done
}

# Get workflow logs (redirect to GitHub)
get_logs() {
    print_color "$BLUE" "📋 Getting Workflow Logs"
    echo "================================================================"
    
    local response=$(call_github_api "actions/runs?per_page=1")
    
    if [ $? -ne 0 ] || [ -z "$response" ]; then
        print_color "$RED" "❌ Failed to fetch latest run"
        return 1
    fi
    
    echo "For detailed logs, visit these links:"
    echo ""
    print_color "$CYAN" "🔗 All Actions: https://github.com/$OWNER/$REPO/actions"
    print_color "$CYAN" "🔗 Latest Run: Check the Actions tab above"
    echo ""
    
    print_color "$YELLOW" "💡 Tip: For command-line logs, install GitHub CLI:"
    echo "winget install GitHub.cli"
    echo "gh run list"
    echo "gh run view [run-id] --log"
}

# Test GitHub connectivity
test_connection() {
    print_color "$BLUE" "🔍 Testing GitHub Connection"
    echo "================================================================"
    
    # Test basic connectivity
    echo "Testing internet connection..."
    if curl -s --head https://github.com >/dev/null; then
        print_color "$GREEN" "✅ Internet connection: OK"
    else
        print_color "$RED" "❌ Internet connection: Failed"
        return 1
    fi
    
    # Test GitHub API
    echo "Testing GitHub API..."
    local response=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
                          -H "Accept: application/vnd.github.v3+json" \
                          https://api.github.com/user)
    
    if [ $? -eq 0 ] && [ -n "$response" ]; then
        print_color "$GREEN" "✅ GitHub API: OK"
        echo "API Response:"
        echo "$response" | head -c 500
        echo "..."
    else
        print_color "$RED" "❌ GitHub API: Failed"
        echo "Check your GITHUB_TOKEN"
        return 1
    fi
    
    # Test repository access
    echo ""
    echo "Testing repository access..."
    local repo_response=$(call_github_api "")
    
    if [ $? -eq 0 ] && [ -n "$repo_response" ]; then
        print_color "$GREEN" "✅ Repository access: OK"
    else
        print_color "$RED" "❌ Repository access: Failed"
        return 1
    fi
}

# Show help
show_help() {
    print_color "$BLUE" "GitHub Actions Monitor for Bash"
    echo "==============================="
    echo ""
    echo "Usage: ./github-actions-bash.sh [command]"
    echo ""
    print_color "$YELLOW" "Commands:"
    echo "  status    - Show latest workflow status"
    echo "  list      - List recent workflow runs"  
    echo "  failed    - Show recent failed runs"
    echo "  monitor   - Monitor in real-time (default 30s)"
    echo "  logs      - Get links to workflow logs"
    echo "  test      - Test GitHub connection"
    echo "  help      - Show this help"
    echo ""
    print_color "$YELLOW" "Setup:"
    echo "  export GITHUB_TOKEN='your_token_here'"
    echo "  Get token from: https://github.com/settings/tokens"
    echo ""
    print_color "$YELLOW" "Examples:"
    echo "  ./github-actions-bash.sh status"
    echo "  ./github-actions-bash.sh monitor"
    echo "  ./github-actions-bash.sh test"
}

# Main script
main() {
    local command=${1:-status}
    
    case "$command" in
        "status")
            check_auth
            show_status
            ;;
        "list")
            check_auth
            list_runs
            ;;
        "failed")
            check_auth
            show_failed
            ;;
        "monitor")
            check_auth
            monitor "${2:-30}"
            ;;
        "logs")
            check_auth
            get_logs
            ;;
        "test")
            check_auth
            test_connection
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_color "$RED" "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"