#!/usr/bin/env bash

for i in {1..100}
do
  echo "Cycle $i"
  python main.py
  rm data/checkpoint
done

echo "The last cycle is $i"
