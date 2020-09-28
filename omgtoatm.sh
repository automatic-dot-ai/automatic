
grep -rl "atm" . | xargs sed -i 's/atm/atm/g'
grep -rl "ATM" . | xargs sed -i 's/ATM/ATM/g'
find . -type f -name "*ATM*" | while read -r file; do
    mv "$file" "${file//ATM/ATM}"
done

