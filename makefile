## Red SN
## ===========================

.DEFAULT_GOAL := help
.PHONY: help run setup


setup: ## set up environment (does not work in make yet)
	conda activate red_sn || (conda env create -f red_sn.yaml --yes && conda activate red_sn)
	
do: run ## Run the main script
	python investigate_data.py


help:  ## Displays this message
		@echo "Commands for the Red SN analysis"
		@echo "Use -B or --always-make to force all dependencies to rerun."
		@echo "Use -i or --ignore-errors to run all sub-commands, even if one fails"
		@echo " "
		@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'