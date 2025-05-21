
for i in {1..10}
do
    echo "Run $i of 20"
    python agent.py
    
    # Rename runlog.jsonl to runlog_$i.jsonl if it exists
    if [ -f runlog.jsonl ]; then
        mv runlog.jsonl "runlog_${i}.jsonl"
        echo "Renamed runlog.jsonl to runlog_${i}.jsonl"
    else
        echo "Warning: runlog.jsonl not found after run $i"
    fi
    
    if [ $i -lt 10 ]; then
        echo "Sleeping for 15 seconds..."
        sleep 15
    fi
done

# find . -type f -name "*.jsonl" -exec sh -c 'mv "$0" "${0%.jsonl}.jsonl2"' {} \;
