# InsightSerenity — Terraform Root Module
# =========================================
# Orchestrates all infrastructure modules.
# Each module is responsible for one layer of the stack.

locals {
  prefix = "${var.project_name}-${var.environment}"
  tags   = { Project = var.project_name, Environment = var.environment }
}

# ── Networking ────────────────────────────────────────────────────────────────
module "networking" {
  source = "./modules/networking"

  prefix             = local.prefix
  vpc_cidr           = var.vpc_cidr
  availability_zones = var.availability_zones
  tags               = local.tags
}

# ── EKS Compute ───────────────────────────────────────────────────────────────
module "compute" {
  source = "./modules/compute"

  prefix             = local.prefix
  vpc_id             = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnet_ids
  eks_version        = var.eks_version
  node_instance_type = var.node_instance_type
  gpu_instance_type  = var.gpu_instance_type
  min_nodes          = var.min_nodes
  max_nodes          = var.max_nodes
  tags               = local.tags
}

# ── PostgreSQL (RDS) ──────────────────────────────────────────────────────────
module "database" {
  source = "./modules/database"

  prefix                   = local.prefix
  vpc_id                   = module.networking.vpc_id
  private_subnet_ids       = module.networking.private_subnet_ids
  eks_security_group_id    = module.compute.node_security_group_id
  db_instance_class        = var.db_instance_class
  db_allocated_storage     = var.db_allocated_storage
  db_max_allocated_storage = var.db_max_allocated_storage
  tags                     = local.tags
}

# ── ElastiCache (Redis) ────────────────────────────────────────────────────────
module "ai" {
  source = "./modules/ai"

  prefix             = local.prefix
  vpc_id             = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnet_ids
  eks_sg_id          = module.compute.node_security_group_id
  redis_node_type    = var.redis_node_type
  tags               = local.tags
}
