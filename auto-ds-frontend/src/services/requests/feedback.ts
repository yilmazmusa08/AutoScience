import {
  IFeedbackRequest,
  IFeedback,
} from "../../containers/Feedback/types/feedback";
import BackendClient from "../BackendClient";
import { FeedbackUrl } from "../constants/urls";

export function createFeedbackRequest(
  payload: IFeedbackRequest,
): Promise<IFeedback> {
  return BackendClient.post(FeedbackUrl.CreateFeedback, payload)
    .then(({ data }) => {
      return Promise.resolve(data);
    })
    .catch((error) => {
      console.error(error);
      return Promise.reject(error);
    });
}
