# InsightSerenity — Terraform Outputs

output "vpc_id" {
  description = "VPC ID"
  value       = module.networking.vpc_id
}

output "eks_cluster_name" {
  description = "EKS cluster name — use with: aws eks update-kubeconfig --name <name>"
  value       = module.compute.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS API server endpoint"
  value       = module.compute.cluster_endpoint
  sensitive   = true
}

output "rds_endpoint" {
  description = "RDS PostgreSQL connection endpoint"
  value       = module.database.endpoint
  sensitive   = true
}

output "rds_database_name" {
  description = "RDS database name"
  value       = module.database.database_name
}

output "redis_endpoint" {
  description = "ElastiCache Redis primary endpoint"
  value       = module.ai.redis_endpoint
  sensitive   = true
}

output "ecr_api_gateway_url" {
  description = "ECR repository URL for the api-gateway image"
  value       = module.compute.ecr_api_gateway_url
}

output "ecr_ai_engine_url" {
  description = "ECR repository URL for the ai-engine image"
  value       = module.compute.ecr_ai_engine_url
}
