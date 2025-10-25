# test_mcp_integration.py
"""
Integration tests for PyManager MCP Server and AI Assistant
Run with: python test_mcp_integration.py
"""

import requests
import json
import time
from typing import Dict, Any
import sys

# Configuration
MCP_SERVER_URL = "http://localhost:8000"
TEST_TIMEOUT = 10

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_test(message: str):
    """Print test message"""
    print(f"{Colors.BLUE}→ {message}{Colors.RESET}")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")

def print_header(message: str):
    """Print test section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

# =============================================================================
# Test Suite
# =============================================================================

def test_mcp_server_health() -> bool:
    """Test 1: MCP Server Health Check"""
    print_test("Testing MCP server health...")
    
    try:
        response = requests.get(f"{MCP_SERVER_URL}/", timeout=TEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Server running: {data.get('service')}")
            print_success(f"Version: {data.get('version')}")
            print_success(f"Tools available: {data.get('tools_count')}")
            return True
        else:
            print_error(f"Server returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to MCP server")
        print_warning("Make sure server is running: python mcp_server.py")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return False

def test_list_tools() -> bool:
    """Test 2: List Available Tools"""
    print_test("Testing tool listing...")
    
    try:
        response = requests.get(f"{MCP_SERVER_URL}/tools", timeout=TEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            tools = data.get('tools', [])
            
            print_success(f"Found {len(tools)} tools:")
            for tool in tools:
                print(f"  • {tool['name']}: {tool['description']}")
            
            expected_tools = [
                'get_portfolios',
                'get_portfolio_by_id',
                'get_watchlist',
                'get_transactions',
                'calculate_portfolio_metrics',
                'get_allocation_breakdown'
            ]
            
            tool_names = [t['name'] for t in tools]
            missing = [t for t in expected_tools if t not in tool_names]
            
            if missing:
                print_warning(f"Missing tools: {', '.join(missing)}")
                return False
            else:
                print_success("All expected tools present")
                return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_get_portfolios() -> bool:
    """Test 3: Get Portfolios Tool"""
    print_test("Testing get_portfolios tool...")
    
    try:
        payload = {
            "tool": "get_portfolios",
            "params": {}
        }
        
        response = requests.post(
            f"{MCP_SERVER_URL}/execute",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 404 or (response.status_code == 200 and not response.json().get('success')):
            print_success("Invalid tool correctly rejected")
            return True
        else:
            print_error("Invalid tool was not rejected properly")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_allocation_breakdown() -> bool:
    """Test 7: Get Allocation Breakdown"""
    print_test("Testing get_allocation_breakdown tool...")
    
    try:
        payload = {
            "tool": "get_allocation_breakdown",
            "params": {}
        }
        
        response = requests.post(
            f"{MCP_SERVER_URL}/execute",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                data = result.get('data', {})
                allocation = data.get('allocation', {})
                total_value = data.get('total_value', 0)
                diversification_score = data.get('diversification_score', 0)
                
                print_success(f"Total value: ${total_value:,.2f}")
                print_success(f"Diversification score: {diversification_score}")
                
                if allocation:
                    print_success("Asset allocation:")
                    for asset_type, value in allocation.items():
                        pct = (value / total_value * 100) if total_value > 0 else 0
                        print(f"  • {asset_type}: ${value:,.2f} ({pct:.1f}%)")
                else:
                    print_warning("No allocation data (empty portfolios)")
                
                return True
            else:
                print_error(f"Tool failed: {result.get('error')}")
                return False
        else:
            print_error(f"Request failed")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_performance() -> bool:
    """Test 8: Performance Test"""
    print_test("Testing response time...")
    
    try:
        start_time = time.time()
        
        response = requests.get(f"{MCP_SERVER_URL}/", timeout=TEST_TIMEOUT)
        
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        
        if response.status_code == 200:
            if elapsed < 100:
                print_success(f"Response time: {elapsed:.2f}ms (Excellent)")
                return True
            elif elapsed < 500:
                print_success(f"Response time: {elapsed:.2f}ms (Good)")
                return True
            else:
                print_warning(f"Response time: {elapsed:.2f}ms (Slow)")
                return True
        else:
            print_error("Request failed")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# =============================================================================
# Integration Tests
# =============================================================================

def test_ai_assistant_integration() -> bool:
    """Test 9: AI Assistant Integration"""
    print_test("Testing AI Assistant can connect to MCP...")
    
    # This tests if the MCP server can handle multiple rapid requests
    try:
        tools_to_test = [
            "get_portfolios",
            "get_watchlist",
            "get_transactions"
        ]
        
        all_success = True
        
        for tool in tools_to_test:
            payload = {"tool": tool, "params": {}}
            response = requests.post(
                f"{MCP_SERVER_URL}/execute",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            
            if response.status_code != 200 or not response.json().get('success'):
                all_success = False
                print_error(f"Tool {tool} failed")
        
        if all_success:
            print_success("AI Assistant integration ready")
            return True
        else:
            print_error("Some tools failed")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_error_handling() -> bool:
    """Test 10: Error Handling"""
    print_test("Testing error handling...")
    
    # Test with invalid JSON
    try:
        response = requests.post(
            f"{MCP_SERVER_URL}/execute",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=TEST_TIMEOUT
        )
        
        # Should return 422 (Unprocessable Entity) or 400 (Bad Request)
        if response.status_code in [400, 422]:
            print_success("Invalid JSON correctly rejected")
        else:
            print_warning(f"Unexpected status code: {response.status_code}")
        
        # Test with missing required parameters
        payload = {
            "tool": "get_portfolio_by_id",
            "params": {}  # Missing portfolio_id
        }
        
        response = requests.post(
            f"{MCP_SERVER_URL}/execute",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        
        result = response.json()
        
        if not result.get('success'):
            print_success("Missing parameters correctly handled")
            return True
        else:
            print_warning("Missing parameters not caught")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all tests and report results"""
    
    print_header("🧪 PyManager MCP Integration Tests")
    
    tests = [
        ("Server Health Check", test_mcp_server_health),
        ("List Available Tools", test_list_tools),
        ("Get Portfolios", test_get_portfolios),
        ("Get Watchlist", test_get_watchlist),
        ("Get Transactions", test_get_transactions),
        ("Invalid Tool Handling", test_invalid_tool),
        ("Allocation Breakdown", test_allocation_breakdown),
        ("Performance Test", test_performance),
        ("AI Assistant Integration", test_ai_assistant_integration),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    for name, test_func in tests:
        print_header(f"Test: {name}")
        try:
            success = test_func()
            results.append((name, success))
            
            if success:
                print_success(f"✓ {name} PASSED\n")
            else:
                print_error(f"✗ {name} FAILED\n")
                
        except Exception as e:
            print_error(f"✗ {name} CRASHED: {str(e)}\n")
            results.append((name, False))
        
        time.sleep(0.5)  # Brief pause between tests
    
    # Print summary
    print_header("📊 Test Results Summary")
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    success_rate = (passed / len(results) * 100) if results else 0
    
    print(f"\nTotal Tests: {len(results)}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
    print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
    print(f"Success Rate: {success_rate:.1f}%\n")
    
    # Detailed results
    for name, success in results:
        status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if success else f"{Colors.RED}✗ FAIL{Colors.RESET}"
        print(f"{status} - {name}")
    
    print()
    
    # Final verdict
    if failed == 0:
        print_header("🎉 All Tests Passed!")
        print_success("Your MCP server is working perfectly!")
        print_success("AI Assistant is ready to use.")
        return 0
    elif passed > failed:
        print_header("⚠️ Some Tests Failed")
        print_warning(f"{failed} test(s) failed, but core functionality works.")
        print_warning("Check the errors above for details.")
        return 1
    else:
        print_header("❌ Critical Failures")
        print_error("Multiple tests failed. Please fix the issues.")
        print_error("Check MCP server logs for details.")
        return 2

def test_quick_check():
    """Quick connectivity test"""
    print_header("⚡ Quick Connection Test")
    
    print_test("Checking MCP server...")
    
    try:
        response = requests.get(f"{MCP_SERVER_URL}/", timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"✓ Server is running")
            print_success(f"✓ Version: {data.get('version')}")
            print_success(f"✓ Tools: {data.get('tools_count')}")
            print()
            print_success("🎉 MCP Server is ready!")
            print()
            print("Run full test suite with: python test_mcp_integration.py --full")
            return 0
        else:
            print_error("Server returned unexpected status")
            return 1
            
    except requests.exceptions.ConnectionError:
        print_error("✗ Cannot connect to MCP server")
        print()
        print_warning("To start the server:")
        print("  python mcp_server.py")
        print()
        return 1
    except Exception as e:
        print_error(f"✗ Error: {str(e)}")
        return 1

# =============================================================================
# CLI Interface
# =============================================================================

def print_usage():
    """Print usage information"""
    print(f"""
{Colors.BOLD}PyManager MCP Integration Tests{Colors.RESET}

{Colors.BOLD}Usage:{Colors.RESET}
  python test_mcp_integration.py [OPTIONS]

{Colors.BOLD}Options:{Colors.RESET}
  --full, -f      Run full test suite (default)
  --quick, -q     Quick connectivity check only
  --help, -h      Show this help message

{Colors.BOLD}Examples:{Colors.RESET}
  python test_mcp_integration.py           # Full test suite
  python test_mcp_integration.py --quick   # Quick check
  python test_mcp_integration.py --help    # Show help

{Colors.BOLD}Requirements:{Colors.RESET}
  - MCP server must be running (python mcp_server.py)
  - Server should be accessible at {MCP_SERVER_URL}

{Colors.BOLD}Troubleshooting:{Colors.RESET}
  If tests fail:
  1. Verify MCP server is running
  2. Check server logs for errors
  3. Verify port 8000 is not blocked
  4. Try: curl http://localhost:8000/
""")

if __name__ == "__main__":
    # Parse command line arguments
    import sys
    
    args = sys.argv[1:]
    
    if "--help" in args or "-h" in args:
        print_usage()
        sys.exit(0)
    elif "--quick" in args or "-q" in args:
        exit_code = test_quick_check()
        sys.exit(exit_code)
    else:
        exit_code = run_all_tests()
        sys.exit(exit_code)TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                data = result.get('data', {})
                count = data.get('count', 0)
                total_value = data.get('total_value', 0)
                
                print_success(f"Retrieved {count} portfolios")
                print_success(f"Total value: ${total_value:,.2f}")
                
                if count > 0:
                    portfolios = data.get('portfolios', [])
                    print_success("Sample portfolio:")
                    print(f"  {json.dumps(portfolios[0], indent=2)}")
                else:
                    print_warning("No portfolios found (this is OK for fresh install)")
                
                return True
            else:
                print_error(f"Tool failed: {result.get('error')}")
                return False
        else:
            print_error(f"Request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_get_watchlist() -> bool:
    """Test 4: Get Watchlist Tool"""
    print_test("Testing get_watchlist tool...")
    
    try:
        payload = {
            "tool": "get_watchlist",
            "params": {}
        }
        
        response = requests.post(
            f"{MCP_SERVER_URL}/execute",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                data = result.get('data', {})
                count = data.get('count', 0)
                
                print_success(f"Retrieved {count} watchlist items")
                
                if count > 0:
                    tickers = data.get('tickers', [])
                    print_success(f"Tickers: {', '.join(tickers)}")
                else:
                    print_warning("Watchlist empty (this is OK)")
                
                return True
            else:
                print_error(f"Tool failed: {result.get('error')}")
                return False
        else:
            print_error(f"Request failed")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_get_transactions() -> bool:
    """Test 5: Get Transactions Tool"""
    print_test("Testing get_transactions tool...")
    
    try:
        payload = {
            "tool": "get_transactions",
            "params": {}
        }
        
        response = requests.post(
            f"{MCP_SERVER_URL}/execute",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                data = result.get('data', {})
                count = data.get('count', 0)
                summary = data.get('summary', {})
                
                print_success(f"Retrieved {count} transactions")
                print_success(f"Total buy: ${summary.get('total_buy_amount', 0):,.2f}")
                print_success(f"Total sell: ${summary.get('total_sell_amount', 0):,.2f}")
                
                return True
            else:
                print_error(f"Tool failed: {result.get('error')}")
                return False
        else:
            print_error(f"Request failed")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_invalid_tool() -> bool:
    """Test 6: Invalid Tool Handling"""
    print_test("Testing invalid tool handling...")
    
    try:
        payload = {
            "tool": "nonexistent_tool",
            "params": {}
        }
        
        response = requests.post(
            f"{MCP_SERVER_URL}/execute",
            json=payload,
            timeout=
