SRC_DIR := ../main
CFLAGS += -std=c11 -Wall -Wextra -Werror -I. -I$(SRC_DIR)

test_fall_detector: test_fall_detector.c $(SRC_DIR)/fall_detector.c $(SRC_DIR)/fall_heuristic.c $(SRC_DIR)/imu_window.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

.PHONY: test clean
test: test_fall_detector
	./test_fall_detector

clean:
	rm -f test_fall_detector
