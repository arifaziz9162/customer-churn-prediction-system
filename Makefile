.PHONY: format format-check lint \
	test-unit test-integration test-serving test \
	check pipeline \
	bento-build bento-image \
	migrate-image migrate \
	docker-up docker-start docker-down docker-ps docker-logs docker-restart docker-clean \
	airflow-up airflow-start airflow-down airflow-ps airflow-logs airflow-restart airflow-clean \
	k8s-namespace k8s-apply k8s-delete k8s-status k8s-clean k8s-logs k8s-rollout k8s-restart \
	k8s-images k8s-services k8s-monitoring k8s-prometheus k8s-grafana k8s-api k8s-predict

# Format Python code.
format:
	black .

# Check formatting without modifying files.
format-check:
	black --check .

# Run lint checks.
lint:
	ruff check .

# Run unit tests.
test-unit:
	pytest tests/unit -v

# Run integration tests.
test-integration:
	pytest tests/integration -v

# Run serving tests.
test-serving:
	pytest tests/serving -v

# Run all tests.
test: test-unit test-integration test-serving

# Run all quality checks.
check: format-check lint test

# Reproduce the DVC pipeline.
pipeline:
	dvc repro

# Build the BentoML package.
bento-build:
	bentoml build -f bentofile.yaml

# Build the BentoML Docker image.
bento-image:
	bentoml containerize customer_churn_service:latest \
		-t customer-churn-service:latest

# Build the database migration image.
migrate-image:
	docker build \
		-f deployment/docker/Dockerfile.migrate \
		-t customer-churn-migrate:latest .

# Run database migrations.
migrate:
	docker run --rm \
		--env-file .env \
		customer-churn-migrate:latest

# Start development services.
docker-up:
	docker compose -f docker-compose.dev.yml up -d --build

# Start services without rebuilding.
docker-start:
	docker compose -f docker-compose.dev.yml up -d

# Check development services.
docker-ps:
	docker compose -f docker-compose.dev.yml ps

# Follow development logs.
docker-logs:
	docker compose -f docker-compose.dev.yml logs -f

# Restart development services.
docker-restart:
	docker compose -f docker-compose.dev.yml restart

# Stop development services.
docker-down:
	docker compose -f docker-compose.dev.yml down

# Remove development resources and volumes.
# WARNING: This may delete persistent database data.
docker-clean:
	docker compose -f docker-compose.dev.yml down -v --remove-orphans

# Start Airflow.
airflow-up:
	docker compose -f docker-compose.airflow.yml up -d --build

# Start Airflow without rebuilding.
airflow-start:
	docker compose -f docker-compose.airflow.yml up -d

# Check Airflow services.
airflow-ps:
	docker compose -f docker-compose.airflow.yml ps

# Follow Airflow logs.
airflow-logs:
	docker compose -f docker-compose.airflow.yml logs -f

# Restart Airflow.
airflow-restart:
	docker compose -f docker-compose.airflow.yml restart

# Stop Airflow.
airflow-down:
	docker compose -f docker-compose.airflow.yml down

# Remove Airflow resources and volumes.
# WARNING: This may delete persisted Airflow metadata.
airflow-clean:
	docker compose -f docker-compose.airflow.yml down -v --remove-orphans

# Create the Kubernetes namespace.
k8s-namespace:
	kubectl create namespace customer-churn --dry-run=client -o yaml | kubectl apply -f -

# Deploy Kubernetes resources.
k8s-apply: k8s-namespace
	kubectl apply -k deployment/k8s/base

# Check Kubernetes resources.
k8s-status:
	kubectl get pods,services,deployments,jobs -n customer-churn

# Delete old migration jobs.
k8s-clean:
	kubectl delete job -n customer-churn \
		-l app=customer-churn-migrate \
		--ignore-not-found

# Check the API rollout.
k8s-rollout:
	kubectl rollout status deployment/customer-churn-service -n customer-churn

# Follow API logs.
k8s-logs:
	kubectl logs -f deployment/customer-churn-service -n customer-churn

# Restart the API deployment.
k8s-restart:
	kubectl rollout restart deployment/customer-churn-service -n customer-churn

# Remove Kubernetes resources.
k8s-delete:
	kubectl delete -k deployment/k8s/base

# List deployed container images.
k8s-images:
	kubectl get deployments -n customer-churn \
		-o jsonpath='{range .items[*]}{.metadata.name}{": "}{range .spec.template.spec.containers[*]}{.image}{" "}{end}{"\n"}{end}'

# List Kubernetes services.
k8s-services:
	kubectl get services -n customer-churn

# Check monitoring resources.
k8s-monitoring:
	kubectl get pods,services,deployments -n customer-churn

# Access Prometheus locally.
k8s-prometheus:
	kubectl port-forward svc/prometheus 9090:9090 -n customer-churn

# Access Grafana locally.
k8s-grafana:
	kubectl port-forward svc/grafana 3001:3000 -n customer-churn

# Access the API locally.
k8s-api:
	kubectl port-forward svc/customer-churn-service 3000:3000 -n customer-churn

# Send a sample churn prediction.
k8s-predict:
	curl -X POST http://127.0.0.1:3000/api/v1/predict \
		-H "Content-Type: application/json" \
		-d '{"gender":"Female","senior_citizen":0,"partner":"Yes","dependents":"No","tenure":12,"phone_service":"Yes","multiple_lines":"No","internet_service":"DSL","online_security":"Yes","online_backup":"No","device_protection":"No","tech_support":"No","streaming_tv":"No","streaming_movies":"No","contract":"Month-to-month","paperless_billing":"Yes","payment_method":"Electronic check","monthly_charges":70.0,"total_charges":850.0}'
