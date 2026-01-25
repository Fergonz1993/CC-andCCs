# Load Testing Suite

This directory contains k6 load testing scripts for the multi-agent coordination system.

## Prerequisites

1. **Install k6**: https://k6.io/docs/get-started/installation/

   ```bash
   # macOS (Homebrew)
   brew install k6

   # Debian/Ubuntu
   sudo gpg -k
   sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
   echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
   sudo apt-get update
   sudo apt-get install k6

   # Windows (Chocolatey)
   choco install k6

   # Docker
   docker pull grafana/k6
   ```

2. **Start the services** you want to test (Option B or C).

## Test Scripts

### Option C: Orchestrator Task Throughput

Tests the Option C orchestrator's task management API.

```bash
# Default run
k6 run option-c-task-throughput.js

# Custom URL
ORCHESTRATOR_URL=http://localhost:8080 k6 run option-c-task-throughput.js

# With more VUs
k6 run --vus 20 --duration 2m option-c-task-throughput.js

# Export results to JSON
k6 run --out json=results-option-c.json option-c-task-throughput.js
```

**Scenarios:**
1. **Task Creation Burst** - Rapid task creation with ramping VUs
2. **Concurrent Claiming** - Multiple agents claiming tasks simultaneously
3. **Sustained Load** - Constant arrival rate for full task lifecycle

### Option B: MCP Server Load

Tests the Option B MCP server's RPC endpoints.

```bash
# Default run
k6 run option-b-mcp-load.js

# Custom URL
MCP_SERVER_URL=http://localhost:3001 k6 run option-b-mcp-load.js

# With custom duration
k6 run --duration 3m option-b-mcp-load.js

# Export results to JSON
k6 run --out json=results-option-b.json option-b-mcp-load.js
```

**Scenarios:**
1. **Agent Lifecycle** - Registration, heartbeats, and work simulation
2. **Task Creation Burst** - Rapid task creation
3. **Concurrent Claiming** - Multiple agents claiming tasks
4. **Batch Operations** - Testing batch API endpoints
5. **Sustained Mixed** - Mixed workload with various operations

## Performance Thresholds

Both scripts define thresholds that must be met for the test to pass:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| `http_req_failed` | < 5% | HTTP error rate |
| `http_req_duration` | p95 < 1000ms | Request latency |
| `task_creation_duration` | p95 < 500ms | Task creation latency |
| `task_claim_duration` | p95 < 300ms | Task claim latency |
| `task_completion_duration` | p95 < 400ms | Task completion latency |
| `task_creation_success` | > 95% | Task creation success rate |
| `task_claim_success` | > 90% | Task claim success rate |
| `rpc_success` | > 95% | RPC call success rate |

## Output Formats

k6 supports multiple output formats:

```bash
# Console summary (default)
k6 run script.js

# JSON output
k6 run --out json=results.json script.js

# CSV output
k6 run --out csv=results.csv script.js

# InfluxDB (for Grafana dashboards)
k6 run --out influxdb=http://localhost:8086/k6 script.js

# Cloud (k6 Cloud)
k6 cloud script.js
```

## Interpreting Results

### Console Output

```
     checks.........................: 95.00% ✓ 950    ✗ 50
     data_received..................: 1.2 MB 20 kB/s
     data_sent......................: 500 kB 8.3 kB/s
     http_req_blocked...............: avg=1.2ms    min=1µs    med=3µs    max=120ms   p(90)=8µs    p(95)=12ms
     http_req_connecting............: avg=900µs    min=0s     med=0s     max=115ms   p(90)=0s     p(95)=11ms
   ✓ http_req_duration..............: avg=45.3ms   min=5ms    med=35ms   max=500ms   p(90)=80ms   p(95)=150ms
   ✓ http_req_failed................: 2.00%  ✓ 20     ✗ 980
     http_reqs......................: 1000   16.67/s
     iteration_duration.............: avg=200ms    min=50ms   med=150ms  max=800ms   p(90)=350ms  p(95)=450ms
     iterations.....................: 1000   16.67/s
   ✓ task_creation_duration.........: avg=40ms     min=5ms    med=30ms   max=200ms   p(90)=70ms   p(95)=100ms
   ✓ task_creation_success..........: 97.00% ✓ 970    ✗ 30
```

### Key Metrics to Monitor

1. **http_req_duration** - Overall request latency
2. **http_req_failed** - Error rate
3. **iterations** - Throughput (requests per second)
4. **checks** - Assertion pass rate
5. **Custom metrics** - Task-specific performance

## CI Integration

Add to your CI pipeline:

```yaml
# GitHub Actions example
load-test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Install k6
      run: |
        sudo gpg -k
        sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
        echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
        sudo apt-get update
        sudo apt-get install k6
    - name: Start services
      run: |
        # Start your services here
        docker-compose up -d
        sleep 10
    - name: Run load tests
      run: |
        k6 run --out json=results.json tests/load/option-b-mcp-load.js
    - name: Upload results
      uses: actions/upload-artifact@v4
      with:
        name: load-test-results
        path: results.json
```

## Customization

### Modifying Thresholds

Edit the `options.thresholds` object in the test scripts:

```javascript
export const options = {
  thresholds: {
    'http_req_duration': ['p(95) < 2000'],  // Increase to 2s
    'task_creation_success': ['rate > 0.80'],  // Lower to 80%
  },
};
```

### Adding Custom Scenarios

```javascript
export const options = {
  scenarios: {
    my_custom_scenario: {
      executor: 'constant-vus',
      vus: 50,
      duration: '5m',
      exec: 'myCustomFunction',
    },
  },
};

export function myCustomFunction() {
  // Your test logic
}
```

## Troubleshooting

### "Connection refused" errors

Ensure the target service is running and accessible:
```bash
curl http://localhost:8080/health  # Option C
curl http://localhost:3001/health  # Option B
```

### High error rates

1. Check service logs for errors
2. Reduce VUs or rate
3. Check resource utilization (CPU, memory)
4. Verify thresholds are realistic for your environment

### Slow performance

1. Check network latency
2. Profile the service for bottlenecks
3. Review database/storage performance
4. Consider horizontal scaling
