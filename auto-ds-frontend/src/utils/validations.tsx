import constants from "../constants";

export const isFileSizeWithinLimit = (fileSizeInBytes: number): boolean => {
  return fileSizeInBytes / 1024 / 1024 <= constants.MaxFileSizeMB;
};
