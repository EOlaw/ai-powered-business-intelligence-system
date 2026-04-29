# InsightSerenity — AI Infrastructure Module (ElastiCache Redis + S3 model storage)

variable "prefix" { type = string }
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "eks_sg_id" { type = string }
variable "redis_node_type" { type = string }
variable "tags" { type = map(string) }

# ── ElastiCache (Redis) ───────────────────────────────────────────────────────

resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.prefix}-redis-subnets"
  subnet_ids = var.private_subnet_ids
  tags       = var.tags
}

resource "aws_security_group" "redis" {
  name   = "${var.prefix}-redis-sg"
  vpc_id = var.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [var.eks_sg_id]
    description     = "Allow EKS to connect to Redis"
  }
  tags = var.tags
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${var.prefix}-redis"
  engine               = "redis"
  node_type            = var.redis_node_type
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  engine_version       = "7.1"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.redis.name
  security_group_ids   = [aws_security_group.redis.id]
  tags                 = var.tags
}

# ── S3 bucket for model weights and training artefacts ────────────────────────

resource "aws_s3_bucket" "models" {
  bucket = "${var.prefix}-model-storage"
  tags   = var.tags
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

output "redis_endpoint" {
  value     = "${aws_elasticache_cluster.redis.cache_nodes[0].address}:6379"
  sensitive = true
}

output "model_bucket_name" {
  value = aws_s3_bucket.models.bucket
}

output "model_bucket_arn" {
  value = aws_s3_bucket.models.arn
}
