TOP_DIR ?= .
SRC_DIR ?= ${TOP_DIR}/src
LOG_DIR ?= ${TOP_DIR}/logs

.PHONY: eval train

eval:
	@echo "Running Pre-Trained Agent"
	cd ${SRC_DIR}; python eval.py
	@echo "Finished."

train:
	@echo "Training Agent"
	cd ${SRC_DIR}; python train.py
	@echo "Finished."

# --- CLEAN ---
.PHONY: clean
.PHONY: clean-logs

clean-logs: ${TOP_DIR}/logs
	rm -r ${LOG_DIR}

clean: clean-logs