TOP_DIR ?= .
SRC_DIR ?= ${TOP_DIR}/src
LOG_DIR ?= ${TOP_DIR}/logs

.PHONY: eval train

eval:
	@echo "Running Pre-Trained Agent"
	python -m src.eval
	@echo "Finished."

train:
	@echo "Training Agent"
	python -m src.train
	@echo "Finished."

# --- CLEAN ---
.PHONY: clean
.PHONY: clean-logs

clean-logs: ${TOP_DIR}/logs
	rm -r ${LOG_DIR}

clean: clean-logs