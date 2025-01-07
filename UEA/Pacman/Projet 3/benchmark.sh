for i in 1 2
do
	for w in large_filter_walls large_filter
	do
		for g in fearless afraid terrified
		do
			python run.py -g $g -ng $i -l $w --seed 42 --nographics | grep -E '^Score'
		done
	done
done
