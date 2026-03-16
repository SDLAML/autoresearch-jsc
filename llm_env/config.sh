SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

## Check if this script is sourced
[[ "$0" != "${SOURCE_PATH}" ]] && echo "Setting vars" || ( echo "Vars script must be sourced." && exit 1) ;
## Determine location of this file
RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"
####################################
###===================
### User Configuration
YOUR_ENV_NAME="venv"
suffix=""


YOUR_ENV_NAME="${YOUR_ENV_NAME}_$SYSTEMNAME$suffix"
#==================
# overwrite $YOUR_ENV_NAME if we want to force acitvate an enviroment
declare -A FORCE_MAP=(
    [force_booster]="juwelsbooster"
    [force_juwels]="juwels"
    [force_jurecadc]="jurecadc"
    [force_hwai]="jureca_hwai"
    [force_gh]="jurecadc_gh"
    [force_jedi]="jedi"
)
for key in "${!FORCE_MAP[@]}"; do
    declare "$key=false"
done

# Parse arguments dynamically
# Parse arguments dynamically
for ARG in "$@"; do
    VAR_NAME="${ARG#--}"  # Remove "--" prefix
    if [[ -n "${FORCE_MAP[$VAR_NAME]}" ]]; then
        FORCED_ENV="${FORCE_MAP[$VAR_NAME]}"  # Store the forced environment
        break  # Stop checking after the first match
    fi
done


if [[ -n "$FORCED_ENV" ]]; then
    YOUR_ENV_NAME="venv_${FORCED_ENV}"
fi




export ENV_NAME="$(basename "$ABSOLUTE_PATH")"             # Default Name of the venv is the directory that contains this file
export ENV_DIR="${ABSOLUTE_PATH}/${YOUR_ENV_NAME}"         # Default location of this VENV is "./venv"
# echo " - Your Virtual Enviroment will be placed at ${ABSOLUTE_PATH}/${YOUR_ENV_NAME}_$SYSTEMNAME"