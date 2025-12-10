output "ecr_repo_url" {
  value = aws_ecr_repository.huk_repo.repository_url
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.ml_cluster.name
}

output "alb_dns_name" {
  value = aws_lb.huk_alb.dns_name
}
