-- Docker 集成测试专用数据库。
-- postgres 镜像只在数据卷为空时执行本目录下的脚本，
-- 因此这里显式建库，让 API 容器有确定的测试库可用。
CREATE DATABASE modelgate_docker_test;
