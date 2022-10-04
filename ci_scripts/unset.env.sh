WORKDIR=$(pwd)
ENV_FILE="${WORKDIR}/${1}"

echo "unset variables from ${ENV_FILE}"
unset $(grep -v '^#' ${ENV_FILE} | sed -E 's/(.*)=.*/\1/' | xargs)
