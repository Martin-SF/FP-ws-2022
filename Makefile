all:
	$(MAKE) -C V46
	$(MAKE) -C V44
	# $(MAKE) -C V60

clean:
	$(MAKE) -C V_basic clean
	$(MAKE) -C V46 clean
	$(MAKE) -C V44 clean
	# $(MAKE) -C v60 clean

.PHONY: all clean
