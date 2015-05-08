lb=40;
ub=90;

for ((i=lb; i<=ub; i+=2))
do
for ((j=i; j<=ub; j+=2))
do

while [ $(jobs | wc -l) -ge 4 ] ; do sleep 1 ; done

./Compete.lnk Testcases/$i.dylib Testcases/$j.dylib result/$i.$j.txt 5 > /dev/null&

done
done

wait

for ((i=lb; i<=ub; i+=2))
do
for ((j=i; j<=ub; j+=2))
do

if ! grep DONE -q result/$i.$j.txt
then
    rm result/$i.$j.txt
else
    sed '$d' result/$i.$j.txt > result/$i.$j.tmp
fi

done
done


cat result/*.*.tmp > result/result.txt
rm result/*.*.t*

python process.py > result/train_data.txt