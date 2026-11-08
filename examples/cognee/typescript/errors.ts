/** An authored validation message that is safe to display without SDK response details. */
export class UserError extends Error {}

export function formatError(error: unknown): string {
  if (error instanceof UserError) return `Error: ${error.message}`;
  return "Error: The example did not complete. Check the configured model routes, input files and storage settings.";
}
