import { IUser } from "../../User/types/user";

export enum FeedbackFormField {
  Feedback = "feedback",
}

export interface IFeedbackRequest {
  [FeedbackFormField.Feedback]: string;
}

export interface IFeedback {
  id: number;
  user: IUser;
  feedback: string;
  created_at: Date;
}
