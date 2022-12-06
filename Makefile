all:
	$(MAKE) -C V46
	$(MAKE) -C V44
	# $(MAKE) -C V60

clean:
	$(MAKE) -C VXX clean
	$(MAKE) -C V46 clean
	$(MAKE) -C V44 clean
	# $(MAKE) -C V60 clean

.PHONY: all clean
