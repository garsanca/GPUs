


CC=gcc

CFLAGS=-O3 -std=c99

PGCC=nvc -fast -acc -gpu=managed -Minfo=accel

LIB=-lm

NAME=nw

default: host gpu

host:
	$(CC) $(CFLAGS) $(NAME).c -o $(NAME).host.exe $(LIB)

gpu: pgi


pgi:
	$(PGCC) $(NAME).c -o $(NAME).pgi.exe $(LIB)

clean: 
	rm -f _ *.exe
