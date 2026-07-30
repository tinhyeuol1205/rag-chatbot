# Engineering Handbook — TechCorp Inc.

## 1. Development Workflow

### 1.1 Git Branching Strategy
TechCorp follows the Git Flow branching model. The main branches are:
- `main`: Production-ready code. Only release branches and hotfixes merge here.
- `develop`: Integration branch for features. All feature branches merge here first.
- `feature/*`: Individual feature branches created from `develop`.
- `hotfix/*`: Emergency fixes created from `main`, merged back to both `main` and `develop`.

Branch naming convention: `feature/JIRA-123-short-description` (e.g., `feature/TC-456-add-user-auth`).

### 1.2 Code Review Process
All code changes require at least 2 approving reviews before merging. Reviewers should check for: (1) correctness and logic, (2) test coverage (minimum 80%), (3) adherence to coding standards, and (4) security vulnerabilities. Reviews should be completed within 24 business hours.

Pull request descriptions must include: a summary of changes, the Jira ticket link, testing steps, and screenshots for UI changes.

### 1.3 CI/CD Pipeline
Our CI/CD pipeline runs on GitHub Actions with the following stages:
1. **Lint**: ESLint for frontend, Ruff for Python, golangci-lint for Go.
2. **Test**: Unit tests, integration tests, and E2E tests (Playwright for frontend).
3. **Build**: Docker image build with multi-stage builds.
4. **Deploy**: Automatic deployment to staging on `develop` merge; manual approval for production deployment from `main`.

Build artifacts are stored in AWS ECR. Deployment manifests are managed via ArgoCD.

### 1.4 Release Process
Releases follow semantic versioning (MAJOR.MINOR.PATCH). A release branch is created from `develop` every two weeks (sprint cadence). The release process includes:
1. Create `release/v1.2.3` branch from `develop`
2. Run full regression test suite
3. Write release notes in CHANGELOG.md
4. Merge to `main` and tag with version
5. Deploy to production after QA sign-off

## 2. Architecture Standards

### 2.1 Microservices Guidelines
Services should be small, focused, and independently deployable. Each service must:
- Own its own database (Database per Service pattern)
- Communicate via REST APIs or message queues (RabbitMQ for async, gRPC for sync)
- Have its own CI/CD pipeline
- Include health check endpoints (`/health` and `/ready`)

Service discovery is handled by Kubernetes DNS. API Gateway (Kong) manages external routing and rate limiting.

### 2.2 Database Standards
- **PostgreSQL**: Primary choice for relational data. Use connection pooling (PgBouncer) in production.
- **Redis**: For caching, session storage, and rate limiting. TTL must be set for all cache keys.
- **MongoDB**: For document storage where schema flexibility is needed. Must use replica sets in production.

All database schemas must be version-controlled using migration tools (Alembic for Python, golang-migrate for Go).

### 2.3 API Design Standards
All REST APIs must follow these conventions:
- Use JSON for request and response bodies
- Use proper HTTP status codes (200, 201, 400, 401, 403, 404, 500)
- Implement pagination for list endpoints using cursor-based pagination
- Version APIs via URL path: `/api/v1/resources`
- Document all endpoints using OpenAPI/Swagger

Rate limiting is enforced at the API Gateway level: 100 requests/minute for authenticated users, 20 requests/minute for anonymous users.

## 3. Monitoring and Observability

### 3.1 Logging Standards
All services must use structured logging (JSON format) with the following required fields: `timestamp`, `level`, `service`, `trace_id`, `message`. Logs are collected by Fluentd and stored in Elasticsearch. Log retention is 30 days for production, 7 days for staging.

Log levels: `DEBUG` (development only), `INFO` (normal operations), `WARN` (potential issues), `ERROR` (failures requiring attention), `FATAL` (service crash).

### 3.2 Metrics and Alerting
Prometheus collects metrics from all services. Key metrics to monitor:
- **Request latency**: p50, p95, p99 (alert if p99 > 2 seconds)
- **Error rate**: 5xx errors / total requests (alert if > 1%)
- **CPU and memory usage**: Alert at 80% utilization
- **Database connection pool**: Alert when pool utilization exceeds 90%

Grafana dashboards are mandatory for every service. PagerDuty is used for on-call alerting with a 15-minute response SLA for critical alerts.

### 3.3 Incident Management
Incidents are classified by severity:
- **SEV1 (Critical)**: Service completely down, affecting all users. Response time: 15 minutes.
- **SEV2 (Major)**: Significant degradation affecting many users. Response time: 30 minutes.
- **SEV3 (Minor)**: Limited impact, workaround available. Response time: 4 hours.

Post-incident reviews (PIRs) are mandatory for SEV1 and SEV2 incidents within 48 hours. PIR documents must include: timeline, root cause analysis, impact assessment, and preventive action items.
