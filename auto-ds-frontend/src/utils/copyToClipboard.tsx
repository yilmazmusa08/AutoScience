import { message } from "antd";
export const copyToClipboard = async (textToCopy: string) => {
  try {
    await navigator.clipboard.writeText(textToCopy);
    message.success("Copied to clipboard!");
  } catch (err) {
    message.error("Failed to copy");
  }
};
