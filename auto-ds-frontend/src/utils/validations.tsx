export const MAX_FILE_SIZE_MB = process.env.MAX_FILE_SIZE_MB || 20;

export const isFileSizeWithinLimit = (fileSizeInBytes: number): boolean => {
  return fileSizeInBytes / 1024 / 1024 <= MAX_FILE_SIZE_MB;
};
