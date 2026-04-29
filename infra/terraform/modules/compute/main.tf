# InsightSerenity — Compute Module (EKS + ECR)
# ==============================================

variable "prefix" { type = string }
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "eks_version" { type = string }
variable "node_instance_type" { type = string }
variable "gpu_instance_type" { type = string }
variable "min_nodes" { type = number }
variable "max_nodes" { type = number }
variable "tags" { type = map(string) }

# ── IAM ───────────────────────────────────────────────────────────────────────

resource "aws_iam_role" "eks_cluster" {
  name = "${var.prefix}-eks-cluster"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "eks.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "eks_cluster" {
  role       = aws_iam_role.eks_cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

resource "aws_iam_role" "eks_nodes" {
  name = "${var.prefix}-eks-nodes"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "nodes_worker" {
  for_each = toset([
    "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
    "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
  ])
  role       = aws_iam_role.eks_nodes.name
  policy_arn = each.value
}

# ── EKS Cluster ───────────────────────────────────────────────────────────────

resource "aws_eks_cluster" "main" {
  name     = "${var.prefix}-cluster"
  version  = var.eks_version
  role_arn = aws_iam_role.eks_cluster.arn

  vpc_config {
    subnet_ids              = var.private_subnet_ids
    endpoint_private_access = true
    endpoint_public_access  = true
  }

  tags       = var.tags
  depends_on = [aws_iam_role_policy_attachment.eks_cluster]
}

# ── General-purpose Node Group ────────────────────────────────────────────────

resource "aws_eks_node_group" "general" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.prefix}-general"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = var.private_subnet_ids
  instance_types  = [var.node_instance_type]

  scaling_config {
    desired_size = var.min_nodes
    min_size     = var.min_nodes
    max_size     = var.max_nodes
  }

  update_config { max_unavailable = 1 }
  tags       = var.tags
  depends_on = [aws_iam_role_policy_attachment.nodes_worker]
}

# ── GPU Node Group (for AI engine) ────────────────────────────────────────────

resource "aws_eks_node_group" "gpu" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.prefix}-gpu"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = [var.private_subnet_ids[0]] # Single AZ for GPU cost control
  instance_types  = [var.gpu_instance_type]

  scaling_config {
    desired_size = 1
    min_size     = 0
    max_size     = 4
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  tags       = var.tags
  depends_on = [aws_iam_role_policy_attachment.nodes_worker]
}

# ── ECR Repositories ──────────────────────────────────────────────────────────

resource "aws_ecr_repository" "api_gateway" {
  name                 = "${var.prefix}/api-gateway"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  tags = var.tags
}

resource "aws_ecr_repository" "ai_engine" {
  name                 = "${var.prefix}/ai-engine"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  tags = var.tags
}

# ── Security Group for nodes ──────────────────────────────────────────────────

data "aws_security_group" "nodes" {
  filter {
    name   = "tag:aws:eks:cluster-name"
    values = [aws_eks_cluster.main.name]
  }
  depends_on = [aws_eks_node_group.general]
}

output "cluster_name" {
  value = aws_eks_cluster.main.name
}

output "cluster_endpoint" {
  value     = aws_eks_cluster.main.endpoint
  sensitive = true
}

output "cluster_ca" {
  value     = aws_eks_cluster.main.certificate_authority[0].data
  sensitive = true
}

output "cluster_token" {
  value     = aws_eks_cluster.main.name
  sensitive = true
}

output "node_security_group_id" {
  value = data.aws_security_group.nodes.id
}

output "ecr_api_gateway_url" {
  value = aws_ecr_repository.api_gateway.repository_url
}

output "ecr_ai_engine_url" {
  value = aws_ecr_repository.ai_engine.repository_url
}
