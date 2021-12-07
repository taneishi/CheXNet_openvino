#!/bin/bash
# Download the 12 tgz files in batches

index=1
for link in $(cat urls.txt) # URLs for the tar.gz files
do
    seq=$(printf "%02d" ${index})
    fn="images_${seq}.tar.gz"
    echo "wget -c -O ${fn} ${link}"
    wget -c -O ${fn} ${link}
    index=$((index+1))
done

echo "Download complete. Please check the checksums"
sha256sum -c SHA256_checksums.txt
