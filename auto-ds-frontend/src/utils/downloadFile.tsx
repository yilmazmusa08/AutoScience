type DataType = Blob | string | Record<string, any>;

export const downloadFile = (data: DataType, filename: string): void => {
  const url = (() => {
    if (data instanceof Blob) {
      return window.URL.createObjectURL(data);
    } else if (typeof data === "string") {
      // Create a Blob from a text string
      return window.URL.createObjectURL(
        new Blob([data], { type: "text/plain" }),
      );
    } else {
      // Create a Blob from a JSON object
      return window.URL.createObjectURL(
        new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
      );
    }
  })();

  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();

  // Clean up
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};
