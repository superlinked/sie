#!/bin/sh
set -e

n8n import:workflow --input=/dusk-webhooks.json
n8n publish:workflow --id=dusk-gate-webhooks

exec n8n start
