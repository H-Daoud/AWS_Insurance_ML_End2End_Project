# ----------------------
# ECR Repository
# ----------------------
locals {
  safe_project_name = replace(var.project_name, "_", "-")
}

resource "aws_ecr_repository" "huk_repo" {
  name         = "huk-rag-chatbot-repo"
  force_delete = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Environment = "Dev"
    Project     = var.project_name
  }
}

# ----------------------
# ECS Cluster
# ----------------------
resource "aws_ecs_cluster" "ml_cluster" {
  name = "${var.project_name}-cluster"
}

# ----------------------
# IAM Role for ECS Task
# ----------------------
resource "aws_iam_role" "ecs_task_execution_role" {
  name = "${var.project_name}-ecs-task-exec-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ----------------------
# ECS Task Definition
# ----------------------
resource "aws_ecs_task_definition" "ml_task" {
  family                   = "${var.project_name}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn

  container_definitions = jsonencode([{
    name      = "ml_container"
    image = "880844766572.dkr.ecr.eu-central-1.amazonaws.com/huk-rag-chatbot-repo:latest"
    essential = true
    portMappings = [{
      containerPort = 8000
      hostPort      = 8000
      protocol      = "tcp"
    }]
  }])
}

# ----------------------
# ECS Service
# ----------------------
resource "aws_ecs_service" "ml_service" {
  name            = "${var.project_name}-service"
  cluster         = aws_ecs_cluster.ml_cluster.id
  task_definition = aws_ecs_task_definition.ml_task.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = data.aws_subnets.default.ids
    assign_public_ip = true
    security_groups = [aws_security_group.sg.id]
  }

  depends_on = [aws_iam_role_policy_attachment.ecs_task_execution_role_policy]
}

# ----------------------
# Security Group
# ----------------------
resource "aws_security_group" "sg" {
  name        = "${var.project_name}-sg"
  description = "Allow HTTP inbound"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ----------------------
# ALB
# ----------------------
resource "aws_lb" "huk_alb" {
  name               = "${local.safe_project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  subnets            = data.aws_subnets.default.ids
  security_groups    = [aws_security_group.sg.id]
}

resource "aws_lb_target_group" "huk_tg" {
  name     = "${local.safe_project_name}-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = data.aws_vpc.default.id
  target_type = "ip"

  health_check {
    path                = "/"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 2
    matcher             = "200-399"
  }
}

resource "aws_lb_listener" "huk_listener" {
  load_balancer_arn = aws_lb.huk_alb.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.huk_tg.arn
  }
}

# ----------------------
# Data sources (default VPC/subnets)
# ----------------------
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}
