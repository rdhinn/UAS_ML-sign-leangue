set -e
apt-get update -qq
apt-get install -y -t trixie libgl1 libglib2.0-0 || apt-get install -y libgl1 libglib2.0-0
