# 先看一眼匹配到谁
pgrep -af "swift deploy .* --served_model_name[ =]judge"

# 再杀：先 TERM，再 KILL 兜底
pkill -f "swift deploy .* --served_model_name[ =]judge" || true
sleep 1
pkill -9 -f "swift deploy .* --served_model_name[ =]judge" || true
