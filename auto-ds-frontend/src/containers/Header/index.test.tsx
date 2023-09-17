import React from "react";
import { render, screen } from "@testing-library/react";
import Header from "./index";

test("renders header", () => {
  render(<Header collapsed={false} setCollapsed={() => {}} />);
  const linkElement = screen.getByText(/AutoVision/i);
  expect(linkElement).toBeInTheDocument();
});
