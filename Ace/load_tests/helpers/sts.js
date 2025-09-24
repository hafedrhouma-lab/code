import http from "k6/http";

// QA environment by default (if not in Kubernetes)
export const ID_URL = __ENV.KUBERNETES_SERVICE_HOST ?
    "http://identity-service.platform-backend.svc.cluster.local" :
    __ENV.STAGE === "prod" ?
        "https://id.talabat.com" :
        "https://id-qa.dhhmena.com";

export function requestAccessToken(clientId = __ENV.STS_CLIENT_ID,
                                   clientSecret = __ENV.STS_CLIENT_SECRET) {
    let response = http.post(
        ID_URL + "/connect/token",
        // k6: "objects will be x-www-form-urlencoded"
        {
            "client_id": clientId,
            "client_secret": clientSecret,
            "grant_type": "client_credentials"
        },
    );

    return response.json();
}
