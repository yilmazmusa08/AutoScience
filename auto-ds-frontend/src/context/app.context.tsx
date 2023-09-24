import {
  createContext,
  useCallback,
  useContext,
  useState,
  useEffect,
} from "react";
import { IPreprocessingRequest } from "../containers/Preprocessing/types/preprocessing";
import {
  IAnalysis,
  IAnalysisRequest,
} from "../containers/Analysis/types/analysis";
import { IModels, IModelsRequest } from "../containers/Models/types/models";
import {
  ILogin,
  IRegister,
  IAuthResponse,
} from "../containers/Authentication/types/authentication";
import { IFeedbackRequest } from "../containers/Feedback/types/feedback";
import { downloadFile } from "../utils/downloadFile";

import { preprocessRequest } from "../services/requests/preprocess";
import { analyzeRequest } from "../services/requests/analyze";
import { modelsRequest } from "../services/requests/models";
import { loginRequest } from "../services/requests/login";
import { registerRequest } from "../services/requests/register";
import { createFeedbackRequest } from "../services/requests/feedback";

import { UserStorage, AuthUser } from "../services/UserStorage";

interface AppContextState {
  analysis: IAnalysis | null;
  models: IModels | null;
  authUser: IAuthResponse | null;
  targetColumns: string[] | undefined;
  targetColumn: string | null;
  file: File | null;
  preprocess: (payload: IPreprocessingRequest) => void;
  analyze: (payload: IAnalysisRequest) => void;
  runModels: (payload: IModelsRequest) => void;
  login: (payload: ILogin) => void;
  logout: () => void;
  register: (payload: IRegister) => void;
  updateTargetColumns: (targetColumns?: string[]) => void;
  updateTargetColumn: (targetColumn: string | null) => void;
  updateFile: (file: File | null) => void;
  createFeedback: (payload: IFeedbackRequest) => void;
  loadingPreprocessing: boolean;
  loadingAnalysis: boolean;
  loadingModels: boolean;
  loadingAuth: boolean;
  loadingFeedback: boolean;
}

const defaultAppContext: AppContextState = {
  analysis: null,
  models: null,
  authUser: null,
  targetColumns: [],
  targetColumn: null,
  file: null,
  preprocess: () => {},
  analyze: () => {},
  runModels: () => {},
  login: () => {},
  logout: () => {},
  register: () => {},
  updateTargetColumns: () => {},
  updateTargetColumn: () => {},
  updateFile: () => {},
  createFeedback: () => {},
  loadingPreprocessing: false,
  loadingAnalysis: false,
  loadingModels: false,
  loadingAuth: false,
  loadingFeedback: false,
};

const useLoadingState = (initialState: boolean) => {
  const [loading, setLoading] = useState(initialState);

  const updateLoading = useCallback((value: boolean) => {
    setLoading(value);
  }, []);

  return [loading, updateLoading] as const;
};

const useAppContext = (props: AppContextState): AppContextState => {
  const [authUser, setAuthUser] = useState(props.authUser);
  const [analysis, setAnalysis] = useState(props.analysis);
  const [models, setModels] = useState(props.models);
  const [targetColumns, setTargetColumns] = useState(props.targetColumns);
  const [targetColumn, setTargetColumn] = useState(props.targetColumn);
  const [file, setFile] = useState(props.file);
  const [loadingPreprocessing, updateLoadingPreprocessing] = useLoadingState(
    props.loadingPreprocessing,
  );
  const [loadingAnalysis, updateLoadingAnalysis] = useLoadingState(
    props.loadingAnalysis,
  );
  const [loadingModels, updateLoadingModels] = useLoadingState(
    props.loadingModels,
  );
  const [loadingAuth, updateLoadingAuth] = useLoadingState(props.loadingAuth);
  const [loadingFeedback, updateLoadingFeedback] = useLoadingState(
    props.loadingFeedback,
  );

  const updateAuthUser = useCallback(
    (authUser: AppContextState["authUser"]) => {
      setAuthUser(authUser);
    },
    [],
  );

  const updateAnalysis = useCallback(
    (analysis: AppContextState["analysis"]) => {
      setAnalysis(analysis);
    },
    [],
  );

  const updateModels = useCallback((models: AppContextState["models"]) => {
    setModels(models);
  }, []);

  const updateTargetColumns = useCallback(
    (targetColumns?: AppContextState["targetColumns"]) => {
      setTargetColumns(targetColumns);
    },
    [],
  );

  const updateTargetColumn = useCallback(
    (targetColumn: AppContextState["targetColumn"]) => {
      setTargetColumn(targetColumn);
    },
    [],
  );

  const updateFile = useCallback(
    (file: AppContextState["file"]) => {
      setFile(file);
      updateTargetColumns(defaultAppContext.targetColumns);
      updateTargetColumn(defaultAppContext.targetColumn);
      updateAnalysis(defaultAppContext.analysis);
      updateModels(defaultAppContext.models);
    },
    [updateTargetColumns, updateTargetColumn, updateAnalysis, updateModels],
  );

  const preprocess = async (payload: IPreprocessingRequest) => {
    updateLoadingPreprocessing(true);
    try {
      const data = await preprocessRequest(payload);
      downloadFile(data, payload.file.name);
    } catch (error) {
      throw error;
    } finally {
      updateLoadingPreprocessing(false);
    }
  };

  const analyze = async (payload: IAnalysisRequest) => {
    updateLoadingAnalysis(true);
    try {
      const data: IAnalysis = await analyzeRequest(payload);
      updateAnalysis(data);
    } catch (error) {
      throw error;
    } finally {
      updateLoadingAnalysis(false);
    }
  };

  const runModels = async (payload: IModelsRequest) => {
    updateLoadingModels(true);
    try {
      const data: IModels = await modelsRequest(payload);
      updateModels(data);
    } catch (error) {
      throw error;
    } finally {
      updateLoadingModels(false);
    }
  };

  const createFeedback = async (payload: IFeedbackRequest) => {
    updateLoadingFeedback(true);
    try {
      await createFeedbackRequest(payload);
    } catch (error) {
      throw error;
    } finally {
      updateLoadingFeedback(false);
    }
  };

  const login = async (payload: ILogin) => {
    updateLoadingAuth(true);
    try {
      const data: IAuthResponse = await loginRequest(payload);
      UserStorage.storeUser(data);
      updateAuthUser(data);
    } catch (error) {
      throw error;
    } finally {
      updateLoadingAuth(false);
    }
  };

  const register = async (payload: IRegister) => {
    updateLoadingAuth(true);
    try {
      const data: IAuthResponse = await registerRequest(payload);
      UserStorage.storeUser(data);
      updateAuthUser(data);
    } catch (error) {
      throw error;
    } finally {
      updateLoadingAuth(false);
    }
  };

  const logout = () => {
    UserStorage.clear();
    updateAuthUser(defaultAppContext.authUser);
    updateFile(defaultAppContext.file);
  };

  useEffect(() => {
    if (UserStorage.isAuthenticated()) {
      const id = UserStorage.getUser(AuthUser.ID);
      const username = UserStorage.getUser(AuthUser.Username);
      const email = UserStorage.getUser(AuthUser.Email);
      const accessToken = UserStorage.getUser(AuthUser.AccessToken);
      const refreshToken = UserStorage.getUser(AuthUser.RefreshToken);
      if (id && username && email && accessToken) {
        updateAuthUser({
          access: accessToken,
          refresh: refreshToken || "",
          user: { pk: parseInt(id, 10), username, email },
        });
      }
    }
  }, [updateAuthUser]);

  return {
    analysis,
    models,
    preprocess,
    analyze,
    runModels,
    login,
    logout,
    register,
    authUser,
    targetColumns,
    targetColumn,
    updateTargetColumns,
    updateTargetColumn,
    file,
    updateFile,
    createFeedback,
    loadingPreprocessing,
    loadingAnalysis,
    loadingModels,
    loadingAuth,
    loadingFeedback,
  };
};

const AppContext = createContext<AppContextState>(defaultAppContext);

export const useApp = (): AppContextState => {
  return useContext(AppContext);
};

export const AppContextConsumer = AppContext.Consumer;

export const AppContextProvider: React.FC<{ children: React.ReactElement }> = ({
  children,
}) => {
  const {
    analysis,
    models,
    preprocess,
    analyze,
    runModels,
    login,
    logout,
    register,
    authUser,
    targetColumns,
    targetColumn,
    updateTargetColumns,
    updateTargetColumn,
    file,
    updateFile,
    createFeedback,
    loadingPreprocessing,
    loadingAnalysis,
    loadingModels,
    loadingAuth,
    loadingFeedback,
  } = useAppContext(defaultAppContext);

  return (
    <AppContext.Provider
      value={{
        analysis,
        models,
        preprocess,
        analyze,
        runModels,
        login,
        logout,
        register,
        authUser,
        targetColumns,
        targetColumn,
        updateTargetColumns,
        updateTargetColumn,
        file,
        updateFile,
        createFeedback,
        loadingPreprocessing,
        loadingAnalysis,
        loadingModels,
        loadingAuth,
        loadingFeedback,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};
