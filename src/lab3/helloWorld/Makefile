


CC=gcc

CFLAGS=-O3 -lm -lrt -std=c99

PGCC=nvc -Minfo -fast -acc -ta=nvidia

NAME=hello

default: host gpu

host:
	$(CC) $(CFLAGS) $(NAME).c -o $(NAME).host.exe

gpu: pgi

pgi:
	$(PGCC) $(NAME).c -o $(NAME).pgi.exe


clean: 
	rm -f __hmpp_*  *.exe
