import { IPreprocessingRequest } from "../../containers/Preprocessing/types/preprocessing";
import BackendClient from "../BackendClient";
import { PreprocessingUrl } from "../constants/urls";

export function preprocessRequest(
  payload: IPreprocessingRequest,
): Promise<boolean> {
  const formData = new FormData();
  formData.append("file", payload.file);
  return BackendClient.post(PreprocessingUrl.Preprocess, formData)
    .then(({ data }) => {
      const url = window.URL.createObjectURL(new Blob([data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", payload.file.name);
      document.body.appendChild(link);
      link.click();
      link?.parentNode?.removeChild(link);
      return Promise.resolve(true);
    })
    .catch((error) => {
      console.log(error);
      return Promise.resolve(false);
    });
}
