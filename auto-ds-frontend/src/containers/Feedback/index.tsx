import { useState, useEffect } from "react";
import {
  Button,
  FloatButton,
  Drawer,
  Space,
  Row,
  Input,
  Form,
  Result,
} from "antd";
import {
  FrownTwoTone,
  MehTwoTone,
  SmileTwoTone,
  LoadingOutlined,
} from "@ant-design/icons";
import { IFeedbackRequest, FeedbackFormField } from "./types/feedback";
import { useApp } from "../../context/app.context";
import "./index.css";

const { TextArea } = Input;

const ResultSuccess: React.FC<{ onClose: () => void }> = ({ onClose }) => (
  <Result
    style={{ padding: 0 }}
    status="success"
    title="Successfully Submitted Feedback!"
    extra={[
      <Button type="primary" onClick={onClose}>
        Close
      </Button>,
    ]}
  />
);

const ResultError: React.FC<{ onClose: () => void }> = ({ onClose }) => (
  <Result
    style={{ padding: 0 }}
    status="error"
    title="Submission Failed"
    //TODO
    //subTitle={error msg}
    extra={[
      <Button type="primary" onClick={onClose}>
        Close
      </Button>,
    ]}
  />
);

const Feedback: React.FC = () => {
  const [form] = Form.useForm();
  const { createFeedback, loadingFeedback } = useApp();
  const [open, setOpen] = useState(false);
  const [isFrownHovered, setFrownHovered] = useState(false);
  const [isMehHovered, setMehHovered] = useState(false);
  const [isSmileHovered, setSmileHovered] = useState(false);
  const [clickedIcon, setClickedIcon] = useState("");
  const [resultVisibility, setResultVisibility] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const showDrawer = () => {
    setOpen(true);
  };

  const onClose = () => {
    setOpen(false);
  };

  useEffect(() => {
    setResultVisibility(false);
  }, [open]);

  const handleIconClick = (iconName: string) => {
    setFrownHovered(iconName === "frown");
    setMehHovered(iconName === "meh");
    setSmileHovered(iconName === "smile");
    setClickedIcon(iconName);
  };

  const submitFeedback = async (payload: IFeedbackRequest) => {
    try {
      await createFeedback(payload);
      setIsSuccess(true);
    } catch (error) {
      setIsSuccess(false);
    } finally {
      setResultVisibility(true);
      form.resetFields();
    }
  };

  return (
    <>
      <FloatButton
        type="primary"
        className="float-button"
        description="Feedback"
        shape="square"
        onClick={showDrawer}
      />
      <Drawer
        contentWrapperStyle={{ height: "360px", margin: "auto 0" }}
        title="Give us your feedback"
        placement="right"
        onClose={onClose}
        open={open}
      >
        {resultVisibility ? (
          isSuccess ? (
            <ResultSuccess onClose={() => setResultVisibility(false)} />
          ) : (
            <ResultError onClose={() => setResultVisibility(false)} />
          )
        ) : (
          <Form form={form} name="feedback" onFinish={submitFeedback}>
            <Row justify="center">
              <Space wrap size="large" align="center">
                <FrownTwoTone
                  twoToneColor={isFrownHovered ? "red" : undefined}
                  className="feedback-icon"
                  onClick={() => handleIconClick("frown")}
                  onMouseEnter={() => setFrownHovered(true)}
                  onMouseLeave={() =>
                    clickedIcon !== "frown" && setFrownHovered(false)
                  }
                />
                <MehTwoTone
                  twoToneColor={isMehHovered ? "#dbd40b" : undefined}
                  className="feedback-icon"
                  onClick={() => handleIconClick("meh")}
                  onMouseEnter={() => setMehHovered(true)}
                  onMouseLeave={() =>
                    clickedIcon !== "meh" && setMehHovered(false)
                  }
                />
                <SmileTwoTone
                  twoToneColor={isSmileHovered ? "#1ccf00" : undefined}
                  className="feedback-icon"
                  onClick={() => handleIconClick("smile")}
                  onMouseEnter={() => setSmileHovered(true)}
                  onMouseLeave={() =>
                    clickedIcon !== "smile" && setSmileHovered(false)
                  }
                />
              </Space>
            </Row>
            <Row className="feedback-input">
              <Form.Item
                name={FeedbackFormField.Feedback}
                rules={[
                  { required: true, message: "Please input your feedback" },
                ]}
                style={{ width: "100%" }}
              >
                <TextArea
                  placeholder="Your feedback"
                  autoSize={{ minRows: 6, maxRows: 6 }}
                  showCount
                  maxLength={500}
                />
              </Form.Item>
            </Row>
            <Row>
              <Form.Item noStyle>
                <Button
                  type="primary"
                  htmlType="submit"
                  disabled={loadingFeedback}
                >
                  {loadingFeedback && <LoadingOutlined />} Submit
                </Button>
              </Form.Item>
            </Row>
          </Form>
        )}
      </Drawer>
    </>
  );
};

export default Feedback;
